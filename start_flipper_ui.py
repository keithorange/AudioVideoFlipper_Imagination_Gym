import numpy as np
from tkinter import Radiobutton, StringVar
from tkinter import Toplevel, PhotoImage
import tkinter as tk

from av_flipper import AudioVideoFlipper, ConfigLoader, FlipperScriptArgs

# Constants for styling
BG_COLOR = '#9A48D0'  # A bright purple color
FG_COLOR = '#FFFFFF'  # White color for text
TR_COLOR = '#C77DF3'  # Trough color for sliders
ACTIVE_BG_COLOR = '#8752A1'  # Slightly darker purple for active background
BUTTON_OUTLINE = "#4CAF50"  # Green color for button outline
BUTTON_FILL = "#4CAF50"  # Green color for button fill

# Function to toggle the advanced settings view


def toggle_advanced_settings():
    for widget in advanced_widgets:
        widget.pack_forget() if not adv_settings.get(
        ) else widget.pack(fill='x', expand=True, pady=2)

# Function to create a labeled frame with a checkbox and slider


def create_labeled_frame(container, text, variable):
    frame = tk.Frame(container, bg=BG_COLOR)
    cb = tk.Checkbutton(frame, text=text, variable=variable, bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR,
                        activebackground=ACTIVE_BG_COLOR, activeforeground=FG_COLOR)
    cb.pack(side='left')
    scale = tk.Scale(frame, from_=0, to_=100, orient='horizontal',
                     bg=BG_COLOR, fg=FG_COLOR, troughcolor=TR_COLOR)
    scale.set(95)
    scale.pack(fill='x', expand=True)
    return frame, scale


# Function to open the transparent window and print all UI variables

# Function to print all variables from the form


def get_flipper_script_args_from_ui(audition_scale, visualization_scale, max_volume_scale, difficulty_scale, adv_settings, overlay_option_var):
    audio_full_on_chance = 100-audition_scale.get()
    video_full_on_chance = 100-visualization_scale.get()
    max_volume = max_volume_scale.get()
    cognitive_load = difficulty_scale.get()
    flip_audio = audition_var.get()
    flip_video = visualization_var.get()

    # Determine overlay settings based on the selected option
    overlay_option = overlay_option_var.get()
    opaque_overlay = overlay_option == 'Opaque Overlay'
    black_overlay = overlay_option == 'Black Overlay'
    # Assume that if neither opaque nor black overlay is selected, the random image is used

    args = FlipperScriptArgs(
        audio_full_on_chance=audio_full_on_chance,
        video_full_on_chance=video_full_on_chance,
        max_volume=max_volume,
        cognitive_load=cognitive_load,
        session_length=0,  # Adjust as needed
        flip_audio=flip_audio,
        flip_video=flip_video,
        opaque_overlay=opaque_overlay,
        black_overlay=black_overlay
    )

    return args


# Function to open the transparent window

# Create the Tkinter application
root = tk.Tk()
root.title('Audio-Video Flipper')
root.configure(bg=BG_COLOR)

# Create the main frame
main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(padx=20, pady=20)

# Title and logo with image
title_frame = tk.Frame(main_frame, bg=BG_COLOR)
# Replace with your image path
logo_image = PhotoImage(file='images/logo_small.png')
logo_label = tk.Label(title_frame, image=logo_image, bg=BG_COLOR)
logo_label.image = logo_image  # Keep a reference to avoid garbage collection
logo_label.pack(side='left')

# If you need to set a size, you would have to resize the image accordingly before this step

title_text = tk.Label(title_frame, text='Audio-Video Flipper',
                      bg=BG_COLOR, fg=FG_COLOR, font=('Helvetica', 16))
title_text.pack(side='left', padx=10)
title_frame.pack(fill='x', expand=True)


# Difficulty slider
difficulty_scale = tk.Scale(main_frame, from_=0, to_=100,
                            orient='horizontal', bg=BG_COLOR, fg=FG_COLOR, troughcolor=TR_COLOR)
difficulty_scale.set(50)
difficulty_label = tk.Label(
    main_frame, text='Difficulty:', bg=BG_COLOR, fg=FG_COLOR)
difficulty_label.pack(fill='x', expand=True)
difficulty_scale.pack(fill='x', expand=True)

# Advanced settings
adv_settings = tk.BooleanVar(value=True)
advanced_widgets = []
show_adv_settings_cb = tk.Checkbutton(main_frame, text='Show Advanced Settings', variable=adv_settings,
                                      bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR,
                                      activebackground=ACTIVE_BG_COLOR, activeforeground=FG_COLOR, command=toggle_advanced_settings)
show_adv_settings_cb.pack()
# Overlay options frame
overlay_frame = tk.LabelFrame(
    main_frame, text='Overlay Options', bg=BG_COLOR, fg=FG_COLOR)
advanced_widgets.append(overlay_frame)

# Variable to hold the current overlay option
overlay_option_var = tk.StringVar(value="Random Image")

# # Radio buttons for overlay options
# overlay_rb1 = tk.Radiobutton(overlay_frame, text='Opaque Overlay', variable=overlay_option_var, value='Opaque Overlay',
#                              bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR, activebackground=ACTIVE_BG_COLOR, activeforeground=FG_COLOR)
overlay_rb2 = tk.Radiobutton(overlay_frame, text='Black Overlay', variable=overlay_option_var, value='Black Overlay',
                             bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR, activebackground=ACTIVE_BG_COLOR, activeforeground=FG_COLOR)
overlay_rb3 = tk.Radiobutton(overlay_frame, text='Random Image', variable=overlay_option_var, value='Random Image',
                             bg=BG_COLOR, fg=FG_COLOR, selectcolor=BG_COLOR, activebackground=ACTIVE_BG_COLOR, activeforeground=FG_COLOR)

# # Packing the radio buttons
# overlay_rb1.pack(anchor='w')
overlay_rb2.pack(anchor='w')
overlay_rb3.pack(anchor='w')


# Audition checkbox and slider
audition_var = tk.BooleanVar(value=True)
audition_frame, audition_scale = create_labeled_frame(
    main_frame, 'Audition', audition_var)
advanced_widgets.append(audition_frame)

# Visualization checkbox and slider
visualization_var = tk.BooleanVar(value=True)
visualization_frame, visualization_scale = create_labeled_frame(
    main_frame, 'Visualization', visualization_var)
advanced_widgets.append(visualization_frame)
# Pack the advanced settings frames if the checkbox is checked
toggle_advanced_settings()


# Max Volume slider and label horizontally aligned
volume_frame = tk.Frame(main_frame, bg=BG_COLOR)
volume_frame.pack(fill='x', expand=True)

max_volume_label = tk.Label(
    volume_frame, text='Max Volume', bg=BG_COLOR, fg=FG_COLOR)
max_volume_label.pack(side='left', fill='x', expand=True)

max_volume = tk.Scale(volume_frame, from_=0, to_=100, orient='horizontal',
                      bg=BG_COLOR, fg=FG_COLOR, troughcolor=TR_COLOR)
max_volume.set(100)  # Default value
max_volume.pack(side='left', fill='x', expand=True)

# Function to create the Start Session button


# CONNECT THE AUDIO-VIDEO FLIPPER CLASS HERE

def load_data_and_start_flipper():
    args = get_flipper_script_args_from_ui(
        audition_scale, visualization_scale, max_volume, difficulty_scale, adv_settings, overlay_option_var)

    # Initialize ConfigLoader class
    config_loader = ConfigLoader('configurations.json', script_args=args)
    # Initialize AudioVideoFlipper class
    flipper = AudioVideoFlipper(
        root, BG_COLOR, BUTTON_FILL, config_loader, args)

    flipper.toggle_flipping()


def create_start_button(container, fn):
    # Adjust with the path to your play button image
    # Ensure this path is correct
    play_img = PhotoImage(file='images/play_green.png')
    button = tk.Button(container, image=play_img, text=" Start",
                       compound="left", fg=BUTTON_FILL, command=fn)
    button.image = play_img  # Keep a reference to avoid garbage collection
    button.pack(pady=10, fill='x')


# Now create the button with an image and text "Start"
create_start_button(main_frame, load_data_and_start_flipper)
# Run the application
root.mainloop()
