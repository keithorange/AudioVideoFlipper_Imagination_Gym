import platform
import subprocess
from tkinter import Toplevel, PhotoImage, Button, Label, Scale, Frame, Tk, messagebox
import tkinter as tk
import json
import os
import random
from typing import List
from PIL import Image, ImageTk
import cv2
import numpy as np
from pydantic import BaseModel
from edit_image import apply_random_effects_cv

# Constants for styling
BG_COLOR = '#9A48D0'
FG_COLOR = '#FFFFFF'
BUTTON_FILL = "#4CAF50"


class FlipperScriptArgs:
    def __init__(self, audio_full_on_chance, video_full_on_chance, max_volume, cognitive_load, session_length, flip_audio, flip_video, opaque_overlay, black_overlay):
        self.audio_full_on_chance = audio_full_on_chance
        self.video_full_on_chance = video_full_on_chance
        self.max_volume = max_volume
        self.cognitive_load = cognitive_load
        self.session_length = session_length
        self.flip_audio = flip_audio
        self.flip_video = flip_video
        self.opaque_overlay = opaque_overlay
        self.black_overlay = black_overlay

    def __str__(self):
        return (f"🔊 Audio Full On Chance: {self.audio_full_on_chance}\n"
                f"🎞️ Video Full On Chance: {self.video_full_on_chance}\n"
                f"🔉 Max Volume: {self.max_volume}\n"
                f"🧠 Cognitive Load: {self.cognitive_load}\n"
                f"⏱️ Session Length: {self.session_length}\n"
                f"🔄 Flip Audio: {self.flip_audio}\n"
                f"🔄 Flip Video: {self.flip_video}\n"
                f"🌫️ Opaque Overlay: {self.opaque_overlay}\n"
                f"⬛ Black Overlay: {self.black_overlay}")


class FlipperConfig(BaseModel):
    audio_state: List[int]
    video_state: List[int]
    volume_level: List[float]
    hold_length: List[float]
    blocking_overlay_opacity: List[float]
    training_notes: str
    cognitive_load: float


def gaussian_weighted_sampling(items, attr, target_x, n, std_dev=10.0):
    """
    Samples n items from a list of FlipperConfig objects based on a Gaussian distribution centered around target_x.
    The function uses the value of the specified attribute in each FlipperConfig object for the Gaussian weighting.

    Args:
    items (list of FlipperConfig objects): List of FlipperConfig objects, each having an attribute that corresponds to a numeric value.
    attr (str): The attribute to access the numeric value in the FlipperConfig objects.
    target_x (float): The target value to center the Gaussian distribution.
    n (int): The number of items to sample.
    std_dev (float): Standard deviation for the Gaussian distribution.

    Returns:
    list of FlipperConfig objects: The selected items (FlipperConfig objects).
    """
    if not items or n > len(items):
        return []

    # Create a deep copy of items for each sampling
    items_copy = items.copy()

    sampled_items = []
    for _ in range(n):
        # Calculate Gaussian weights based on the value of the specified attribute
        weights = [np.exp(-((getattr(item, attr) - target_x)
                          ** 2) / (2 * std_dev ** 2)) for item in items_copy]

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Random weighted selection
        sampled_item = random.choices(
            items_copy, weights=normalized_weights, k=1)[0]
        sampled_items.append(sampled_item)

        # Remove the sampled item from the list
        items_copy.remove(sampled_item)

    return sampled_items


class ConfigLoader:
    def __init__(self, config_path, script_args):
        with open(config_path, 'r') as file:
            self.configs = [FlipperConfig(**config)
                            for config in json.load(file)]
        self.script_args = script_args

    def get_next_config(self):
        args = self.script_args
        cognitive_load = args.cognitive_load
        selected_config = gaussian_weighted_sampling(
            self.configs, 'cognitive_load', target_x=cognitive_load, n=1, std_dev=1.0)[0]

        # Here selected_config is already a FlipperConfig object
        # Apply script arguments to the configuration
        selected_config = self.apply_script_args_to_config(
            selected_config)

        return selected_config

    def apply_script_args_to_config(self, config: FlipperConfig):
        if self.script_args.opaque_overlay:
            config.blocking_overlay_opacity = [
                1.0] * len(config.blocking_overlay_opacity)
        config = self.apply_full_on_islands(config, self.script_args)
        config = self.scale_volume_levels(config, self.script_args.max_volume)
        return config

    def apply_full_on_islands(self, config: FlipperConfig, args: FlipperScriptArgs):
        num_responses = len(config.audio_state)

        def apply_full_on_range(start, end, setting_fn):
            for i in range(start, end):
                setting_fn(i)

        def apply_audio_full_on(i):
            config.audio_state[i] = 1
            config.volume_level[i] = args.max_volume

        def apply_video_full_on(i):
            config.video_state[i] = 1
            config.blocking_overlay_opacity[i] = 0.0

        if args.audio_full_on_chance > 0:
            start_index = random.randint(
                0, num_responses - int(num_responses * (args.audio_full_on_chance / 100)))
            end_index = start_index + \
                int(num_responses * (args.audio_full_on_chance / 100))
            apply_full_on_range(start_index, end_index, apply_audio_full_on)

        if args.video_full_on_chance > 0:
            start_index = random.randint(
                0, num_responses - int(num_responses * (args.video_full_on_chance / 100)))
            end_index = start_index + \
                int(num_responses * (args.video_full_on_chance / 100))
            apply_full_on_range(start_index, end_index, apply_video_full_on)

        return config

    def scale_volume_levels(self, config: FlipperConfig, max_volume: float):
        config.volume_level = [scale_volume_level(
            volume, max_volume) for volume in config.volume_level]
        return config


def scale_volume_level(volume_level, max_volume):
    return float(min(volume_level, max_volume))


class AudioVideoFlipper:
    def __init__(self, root, bg_color, button_fill, config_loader, args):
        self.root = root
        self.bg_color = bg_color
        self.button_fill = button_fill
        self.config_loader = config_loader
        self.flipper_window = None
        self.running = False
        self.args = args

    def confirm_exit(self, event):
        response = messagebox.askyesno(
            "Stop Flipper", "Do you wish to stop the audio-video flipper?")
        if response:
            self.stop_flipping()

    def set_system_volume(self, volume):
        # volume is between 0.0 and 100.0
        
        system_name = platform.system()
        if system_name == "Windows":
            # For Windows, using nircmd.exe, scale volume from 0-100 to 0-65535
            volume_scaled = int((65535 * volume) / 100)
            subprocess.call(["nircmd.exe", "setsysvolume", str(volume_scaled)])
            
        elif system_name == "Darwin":  # macOS
            # For macOS, volume can be set directly in the 0-100 range
            subprocess.call(["osascript", "-e", f"set volume output volume {volume}"])
            
        elif system_name == "Linux":
            # For Linux, the amixer command also expects volume in the 0-100 range
            subprocess.call(["amixer", "-D", "pulse", "sset", "Master", f"{volume}%"])
            
        else:
            print("Unsupported operating system for volume control")


    def toggle_flipping(self):
        if self.running:
            self.stop_flipping()
        else:
            self.start_flipping()

    def start_flipping(self):
        if self.running:
            return

        self.running = True
        self.root.withdraw()
        self.open_flipper_window()  # Ensure the flipper window is created before proceeding
        self.run_flipping()

    def stop_flipping(self):
        self.set_system_volume(self.args.max_volume)
        self.running = False
        if self.flipper_window:
            self.flipper_window.destroy()
            self.flipper_window = None
        # Make the menu window fully opaque again
        self.root.deiconify()  # Show the menu window again
        print("Finished flipping.")

    def run_flipping(self):
        if not self.running:
            return

        # Initialize configuration set and index on first run
        if not hasattr(self, 'current_config') or not hasattr(self, 'current_index'):
            self.current_config = self.config_loader.get_next_config()
            self.current_index = 0

        # Check if all configurations in the set have been processed
        if self.current_index >= len(self.current_config.blocking_overlay_opacity):
            self.current_config = self.config_loader.get_next_config()
            self.current_index = 0

        # Fetch current configuration details
        opacity = self.current_config.blocking_overlay_opacity[self.current_index]
        hold_length = self.current_config.hold_length[self.current_index]
        volume = self.current_config.volume_level[self.current_index]

        # Set system volume and transparency for the current configuration
        self.set_system_volume(volume)
        self.set_transparency(opacity)

        # Update the image for each configuration
        self.update_image()

        # Increment index for next configuration in the set
        self.current_index += 1

        # Schedule the next run
        self.root.after(int(hold_length * 1000), self.run_flipping)


    def update_image(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        if self.args.black_overlay:
            black_image = np.zeros(
                (screen_height, screen_width, 3), dtype=np.uint8)
            pil_image = Image.fromarray(black_image)
        else:
            selected_image = random.choice(
                os.listdir('images/blocking_images'))
            cv_image = cv2.imread(f'images/blocking_images/{selected_image}')
            cv_image = apply_random_effects_cv(cv_image)
            pil_image = Image.fromarray(
                cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            pil_image = pil_image.resize(
                (screen_width, screen_height), Image.Resampling.LANCZOS)

        photo_image = ImageTk.PhotoImage(pil_image)
        label = tk.Label(self.flipper_window, image=photo_image)
        label.image = photo_image  # Keep a reference
        label.pack(fill='both', expand=True)

    def set_transparency(self, opacity):
        self.flipper_window.attributes(
            "-alpha", opacity if not self.args.opaque_overlay else 1.0)

    def open_flipper_window(self):
        if self.flipper_window:
            self.flipper_window.destroy()

        self.root.attributes("-alpha", 1)

        self.flipper_window = Toplevel(self.root)
        #self.flipper_window.overrideredirect(True)
        #self.flipper_window.attributes("-topmost", True)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.flipper_window.geometry(f"{screen_width}x{screen_height}+0+0")

        self.update_image()
        # Assuming initial_opacity is defined elsewhere
        self.initial_opacity = 0.88
        self.set_transparency(self.initial_opacity)

        self.flipper_window.bind("<Button-1>", self.confirm_exit)


if __name__ == '__main__':

    # Load configurations
    args = FlipperScriptArgs(audio_full_on_chance=10, video_full_on_chance=10,
                             max_volume=1.0, cognitive_load=0.5, session_length=60, flip_audio=True, flip_video=True, opaque_overlay=False, black_overlay=False)
    config_loader = ConfigLoader('configurations.json', script_args=args)

    # Create the Tkinter application
    root = Tk()
    root.title('Audio-Video Flipper')
    root.configure(bg=BG_COLOR)

    # Create the main frame
    main_frame = Frame(root, bg=BG_COLOR)
    main_frame.pack(padx=20, pady=20)

    # Initialize AudioVideoFlipper class
    flipper = AudioVideoFlipper(
        root, BG_COLOR, BUTTON_FILL, config_loader, args)

    # Create the Start/Stop Flipper button
    start_button = Button(main_frame, text="Start/Stop Flipping",
                          bg=BUTTON_FILL, command=flipper.toggle_flipping)
    start_button.pack(pady=10)

    # Run the application
    root.mainloop()
