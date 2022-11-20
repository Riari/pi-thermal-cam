#!/usr/bin/env python3

import time, board, busio, traceback
import numpy as np
import adafruit_mlx90640
import datetime as dt
import cv2
import logging
import cmapy
from scipy import ndimage

# Initialise logging
logging.basicConfig(
    filename = './cam_log.log',
    filemode = 'a',
    format = '%(asctime)s %(levelname)-8s [%(filename)s:%(name)s:%(lineno)d] %(message)s',
    level = logging.WARNING,
    datefmt = '%d-%b-%y %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Camera:
    # Dependencies
    i2c = None
    mlx = None

    # Image processing
    use_smoothing_filter: bool = True
    colormap_list = ['jet', 'bwr', 'seismic', 'coolwarm', 'PiYG_r', 'tab10', 'tab20', 'gnuplot2', 'brg']
    colormap_index = 0
    interpolation_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, 5, 6]
    interpolation_labels_list = ['Nearest', 'Inter Linear', 'Inter Area', 'Inter Cubic', 'Inter Lanczos4', 'Pure Scipy', 'Scipy/CV2 Mixed']
    interpolation_index = 3
    is_current_frame_processed = False  # Tracks if the current processed image matches the current raw image
    blank_image = np.zeros((24 * 32,))
    timestamp = time.time()

    # General state
    temp_min = None
    temp_max = None
    file_saved_notification_started_at = None

    def __init__(self):
        self.__init_therm_cam()

    def __init_therm_cam(self):
        """Initialize the thermal camera"""
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency = 800000)  # setup I2C
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)  # begin MLX90640 with I2C comm
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ  # set refresh rate
        time.sleep(0.1)

    def __colormap(self):
        return self.colormap_list[self.colormap_index]
    
    def __interpolation(self):
        return self.interpolation_list[self.interpolation_index]
    
    def __interpolation_label(self):
        return self.interpolation_labels_list[self.interpolation_index]

    def generate_frame_image(self):
        """Pull raw temperature data, process it to an image, and update image text"""
        image = self.__pull_raw_image()
        image = self.__process_raw_image(image)
        image = self.__add_image_text(image)
        self.timestamp = time.time()  # Update timestamp for this frame
        self.current_frame_processed = True
        return image

    def cycle_colormap(self):
        self.colormap_index += 1
        if self.colormap_index >= len(self.colormap_list):
            self.colormap_index = 0

    def cycle_interpolation(self):
        self.interpolation_index += 1
        if self.interpolation_index >= len(self.interpolation_list):
            self.interpolation_index = 0

    def __pull_raw_image(self):
        """Get one pull of the raw image data, converting temp units if necessary"""
        image = self.blank_image
        try:
            self.mlx.getFrame(image)  # read mlx90640
            self.temp_min = np.min(image)
            self.temp_max = np.max(image)
            image = self.__temps_to_rescaled_uints(image, self.temp_min, self.temp_max)
            self.current_frame_processed = False  # Note that the newly updated raw frame has not been processed
        except ValueError:
            print("Math error; continuing...")
            image = self.blank_image
            logger.info(traceback.format_exc())
        except OSError:
            print("IO Error; continuing...")
            image = self.blank_image
            logger.info(traceback.format_exc())

        return image

    def __temps_to_rescaled_uints(self, f, Tmin, Tmax):
        """Convert temperatures to pixels on image"""
        f = np.nan_to_num(f)
        norm = np.uint8((f - Tmin) * 255 / (Tmax - Tmin))
        norm.shape = (24, 32)
        return norm

    def __process_raw_image(self, image):
        """Process the raw temp data to a colored image. Filter if necessary"""
        processed_image = image
        # Can't apply colormap before ndimage, so reversed in first two options, even though it seems slower
        if self.interpolation_index == 5:  # Scale via scipy only - slowest but seems higher quality
            processed_image = ndimage.zoom(image, 25)  # interpolate with scipy
            processed_image = cv2.applyColorMap(processed_image, cmapy.cmap(self.__colormap()))
        elif self.interpolation_index == 6:  # Scale partially via scipy and partially via cv2 - mix of speed and quality
            processed_image = ndimage.zoom(image, 10)  # interpolate with scipy
            processed_image = cv2.applyColorMap(processed_image, cmapy.cmap(self.__colormap()))
            processed_image = cv2.resize(processed_image, (800, 600), interpolation = cv2.INTER_CUBIC)
        else:
            processed_image = cv2.applyColorMap(image, cmapy.cmap(self.__colormap()))
            processed_image = cv2.resize(processed_image, (800, 600), interpolation = self.__interpolation())
        processed_image = cv2.flip(processed_image, 1)
        if self.use_smoothing_filter:
            processed_image = cv2.bilateralFilter(processed_image, 15, 80, 80)

        return processed_image

    def __add_image_text(self, image):
        """Set image text content"""
        x = 30
        y = 30
        shiftY = 22
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = .6
        color = (255, 255, 255)
        thickness = 2

        lines = [
            f'Tmin: {self.temp_min:+.1f}',
            f'Tmax: {self.temp_max:+.1f}',
            f'Interp.: {self.__interpolation_label()}',
            f'Colormap: {self.__colormap()}',
            f'Filtered: {self.use_smoothing_filter}',
            f'FPS: {1 / (time.time() - self.timestamp):.1f}'
        ]

        for line in lines:
            cv2.putText(image, line, (x, y), font, scale, color, thickness)
            y += shiftY

        return image

class CameraWindow:
    class Button:
        text: str
        start: list
        end: list
        callback = None

    camera: Camera
    title = 'Thermal Camera'
    is_exit_requested = False
    current_image = None
    queue_save_snapshot = False
    snapshot_width = 1200
    snapshot_height = 900
    snapshot_output_path = './thermal_snapshots/'
    snapshot_saved_at = None
    buttons = []

    def __init__(self, camera: Camera):
        self.camera = camera
        # TODO: make this nicer
        self.__add_button('Save', (0, 550), 80, 50, self.__queue_save_snapshot)
        self.__add_button('Colormap', (90, 550), 130, 50, self.camera.cycle_colormap)
        self.__add_button('Interpolation', (230, 550), 150, 50, self.camera.cycle_interpolation)
        self.__add_button('Exit', (390, 550), 70, 50, self.__exit)

    def start(self):
        """Start the main loop"""

        while not self.is_exit_requested:
            try:
                self.current_image = self.camera.generate_frame_image()

                if self.queue_save_snapshot:
                    self.__save_snapshot()
                else:
                    self.__draw_buttons()

                self.__update_window()
                self.__process_events()
            except RuntimeError as error:
                if error.message == 'Too many retries':
                    print("Too many retries error caught, potential I2C baudrate issue: continuing...")
                    continue
                raise

    def __add_button(self, text: str, start: list, width: int, height: int, callback):
        button = CameraWindow.Button()
        button.text = text
        button.start = start
        button.end = (start[0] + width, start[1] + height)
        button.callback = callback
        self.buttons.append(button)

    def __update_window(self):
        # For a brief period after saving, display saved notification
        if self.snapshot_saved_at is not None:
            if (time.monotonic() - self.snapshot_saved_at) < 1:
                color = (255, 255, 255)
                cv2.putText(self.current_image, 'Snapshot Saved!', (30, 500), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
            else:
                self.snapshot_saved_at = None

        cv2.namedWindow(self.title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(self.title, self.snapshot_width, self.snapshot_height)
        cv2.imshow(self.title, self.current_image)

    def __draw_buttons(self):
        padding = (20, 30)
        color_background = (200, 200, 200)
        color_foreground = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .6
        font_thickness = 2

        for button in self.buttons:
            text_position = (button.start[0] + padding[0], button.start[1] + padding[1])
            cv2.rectangle(self.current_image, button.start, button.end, color_background, -1)
            cv2.putText(self.current_image, button.text, text_position, font, font_scale, color_foreground, font_thickness)

    def __queue_save_snapshot(self):
        """Queue a snapshot to save on the next frame (used for omitting the buttons)"""
        self.queue_save_snapshot = True

    def __save_snapshot(self):
        """Save the current frame as a snapshot to the output folder."""
        filename = self.snapshot_output_path + 'snapshot_' + dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.jpg'
        cv2.imwrite(filename, self.current_image)
        self.snapshot_saved_at = time.monotonic()
        print('Snapshot saved: ', filename)
        self.queue_save_snapshot = False

    def __process_events(self):
        cv2.setMouseCallback(self.title, self.__process_click_input)

        key = cv2.waitKey(1) & 0xFF

        if key == 27: # Escape
            self.__exit()
    
    def __process_click_input(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP: return

        for button in self.buttons:
            if x > button.start[0] and y > button.start[1] and x < button.end[0] and y < button.end[1]:
                button.callback()
    
    def __exit(self):
        cv2.destroyAllWindows()
        self.is_exit_requested = True
        print("Exited thermal camera")

def main():
    window = CameraWindow(Camera())
    window.start()

if __name__ == "__main__":
    main()
