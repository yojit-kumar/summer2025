#!/usr/bin/env python3

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
import json
import subprocess
import psutil
import os
import threading
import time
from pathlib import Path

class GruvboxMenu:
    def __init__(self):
        # Gruvbox color scheme
        self.colors = {
            'bg': '#282828',
            'bg_soft': '#32302f',
            'bg_hard': '#1d2021',
            'fg': '#ebdbb2',
            'red': '#cc241d',
            'green': '#98971a',
            'yellow': '#d79921',
            'blue': '#458588',
            'purple': '#b16286',
            'aqua': '#689d6a',
            'orange': '#d65d0e',
            'gray': '#928374',
            'light_gray': '#a89984'
        }
        
        # CSS styling
        self.css_provider = Gtk.CssProvider()
        self.css_provider.load_from_data(self.get_css().encode())
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), 
            self.css_provider, 
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        
        self.window = None
        self.setup_window()
        
    def get_css(self):
        return f"""
        .menu-window {{
            background-color: {self.colors['bg']};
            border: 2px solid {self.colors['gray']};
            border-radius: 10px;
        }}
        
        .section {{
            background-color: {self.colors['bg_soft']};
            border-radius: 8px;
            margin: 8px;
            padding: 12px;
        }}
        
        .top-section {{
            background-color: {self.colors['bg_hard']};
            min-height: 80px;
        }}
        
        .middle-section {{
            min-height: 200px;
        }}
        
        .bottom-section {{
            min-height: 60px;
        }}
        
        .username-label {{
            color: {self.colors['fg']};
            font-family: "Open Sans";
            font-size: 16px;
            font-weight: bold;
        }}
        
        .avatar-frame {{
            border: 2px solid {self.colors['aqua']};
            border-radius: 25px;
        }}
        
        .app-button {{
            background-color: {self.colors['bg_hard']};
            border: 1px solid {self.colors['gray']};
            border-radius: 8px;
            padding: 8px;
            margin: 4px;
            min-width: 60px;
            min-height: 60px;
        }}
        
        .app-button:hover {{
            background-color: {self.colors['blue']};
            border-color: {self.colors['light_gray']};
        }}
        
        .tool-button {{
            background-color: {self.colors['bg']};
            border: 1px solid {self.colors['orange']};
            border-radius: 6px;
            padding: 6px;
            margin: 2px;
            min-width: 45px;
            min-height: 45px;
        }}
        
        .tool-button:hover {{
            background-color: {self.colors['orange']};
        }}
        
        .usage-bar {{
            background-color: {self.colors['bg_hard']};
            border-radius: 4px;
            margin: 4px 0;
            padding: 4px;
        }}
        
        .usage-label {{
            color: {self.colors['fg']};
            font-family: "Open Sans";
            font-size: 12px;
        }}
        
        .progress-bar {{
            background-color: {self.colors['bg']};
            border-radius: 2px;
        }}
        
        .progress-bar progress {{
            background-color: {self.colors['green']};
        }}
        """
    
    def setup_window(self):
        self.window = Gtk.Window()
        self.window.set_decorated(False)
        
        # Fix for Wayland - use NORMAL window type instead of POPUP_MENU
        self.window.set_type_hint(Gdk.WindowTypeHint.NORMAL)
        
        # Alternative approach for Wayland compatibility
        self.window.set_skip_taskbar_hint(True)
        self.window.set_skip_pager_hint(True)
        self.window.set_keep_above(True)
        
        # Set window to be on top of all other windows
        self.window.set_accept_focus(True)
        
        # Set size
        self.window.set_default_size(350, 500)
        self.window.set_resizable(False)
        
        # Add CSS class
        self.window.get_style_context().add_class('menu-window')
        
        # Connect close events
        self.window.connect("focus-out-event", self.on_focus_out)
        self.window.connect("key-press-event", self.on_key_press)
        self.window.connect("button-press-event", self.on_button_press)
        
        # Main container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.window.add(main_box)
        
        # Top section
        self.create_top_section(main_box)
        
        # Middle section
        self.create_middle_section(main_box)
        
        # Bottom section
        self.create_bottom_section(main_box)
        
        # Start hardware monitoring
        self.start_hardware_monitoring()
    
    def on_button_press(self, widget, event):
        # Handle clicks outside the menu to close it
        if event.type == Gdk.EventType.BUTTON_PRESS:
            allocation = widget.get_allocation()
            if (event.x < 0 or event.x > allocation.width or 
                event.y < 0 or event.y > allocation.height):
                self.hide_menu()
        return False
        
    def create_top_section(self, parent):
        section_frame = Gtk.Frame()
        section_frame.get_style_context().add_class('section')
        section_frame.get_style_context().add_class('top-section')
        
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        section_frame.add(hbox)
        
        # Avatar
        avatar_frame = Gtk.Frame()
        avatar_frame.get_style_context().add_class('avatar-frame')
        avatar_frame.set_size_request(50, 50)
        
        # Try to load avatar image
        try:
            avatar_path = os.path.expanduser("~/.config/waybar/avatar.png")
            if os.path.exists(avatar_path):
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(avatar_path, 46, 46, True)
                avatar_image = Gtk.Image.new_from_pixbuf(pixbuf)
            else:
                avatar_image = Gtk.Image.new_from_icon_name("user-info", Gtk.IconSize.LARGE_TOOLBAR)
        except:
            avatar_image = Gtk.Image.new_from_icon_name("user-info", Gtk.IconSize.LARGE_TOOLBAR)
        
        avatar_frame.add(avatar_image)
        hbox.pack_start(avatar_frame, False, False, 0)
        
        # Username
        username = os.getenv('USER', 'User')
        username_label = Gtk.Label(label=username)
        username_label.get_style_context().add_class('username-label')
        username_label.set_halign(Gtk.Align.START)
        username_label.set_valign(Gtk.Align.CENTER)
        hbox.pack_start(username_label, True, True, 0)
        
        parent.pack_start(section_frame, False, False, 0)
    
    def create_middle_section(self, parent):
        section_frame = Gtk.Frame()
        section_frame.get_style_context().add_class('section')
        section_frame.get_style_context().add_class('middle-section')
        
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        section_frame.add(hbox)
        
        # Left section - App shortcuts (2x2 grid)
        left_frame = Gtk.Frame()
        left_frame.set_label("Applications")
        left_grid = Gtk.Grid()
        left_grid.set_row_spacing(8)
        left_grid.set_column_spacing(8)
        
        # App shortcuts
        apps = [
            ("Firefox", "firefox", "firefox"),
            ("Terminal", "terminal", "kitty"),
            ("Files", "folder", "thunar"),
            ("Code", "text-editor", "code")
        ]
        
        for i, (name, icon, command) in enumerate(apps):
            button = Gtk.Button()
            button.get_style_context().add_class('app-button')
            button.set_tooltip_text(name)
            
            # Try to load custom icon or fall back to system icon
            try:
                icon_path = os.path.expanduser(f"~/.config/waybar/icons/{icon}.png")
                if os.path.exists(icon_path):
                    pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(icon_path, 32, 32, True)
                    image = Gtk.Image.new_from_pixbuf(pixbuf)
                else:
                    image = Gtk.Image.new_from_icon_name(icon, Gtk.IconSize.LARGE_TOOLBAR)
            except:
                image = Gtk.Image.new_from_icon_name(icon, Gtk.IconSize.LARGE_TOOLBAR)
            
            button.add(image)
            button.connect("clicked", self.launch_app, command)
            
            row = i // 2
            col = i % 2
            left_grid.attach(button, col, row, 1, 1)
        
        left_frame.add(left_grid)
        hbox.pack_start(left_frame, True, True, 0)
        
        # Right section - Tools (4x4 grid)
        right_frame = Gtk.Frame()
        right_frame.set_label("Tools")
        right_grid = Gtk.Grid()
        right_grid.set_row_spacing(4)
        right_grid.set_column_spacing(4)
        
        tools = [
            ("Color Picker", "colorhunt", "hyprpicker -a"),
            ("Screenshot", "camera-photo", "grim -g \"$(slurp)\" ~/Pictures/screenshot.png"),
            ("Screen Record", "media-record", "wf-recorder -g \"$(slurp)\" -f ~/Videos/recording.mp4"),
            ("Screenshots", "folder-pictures", "thunar ~/Pictures"),
            ("Videos", "folder-videos", "thunar ~/Videos"),
            ("Settings", "preferences-system", "hyprctl dispatch exec \"gtk-launch org.gnome.Settings\""),
            ("Power", "system-shutdown", "wlogout"),
            ("Info", "dialog-information", "neofetch")
        ]
        
        for i, (name, icon, command) in enumerate(tools[:8]):
            button = Gtk.Button()
            button.get_style_context().add_class('tool-button')
            button.set_tooltip_text(name)
            
            try:
                image = Gtk.Image.new_from_icon_name(icon, Gtk.IconSize.BUTTON)
            except:
                image = Gtk.Image.new_from_icon_name("application-x-executable", Gtk.IconSize.BUTTON)
            
            button.add(image)
            if command:
                button.connect("clicked", self.launch_command, command)
            
            row = i // 4
            col = i % 4
            right_grid.attach(button, col, row, 1, 1)
        
        right_frame.add(right_grid)
        hbox.pack_start(right_frame, True, True, 0)
        
        parent.pack_start(section_frame, True, True, 0)
    
    def create_bottom_section(self, parent):
        section_frame = Gtk.Frame()
        section_frame.get_style_context().add_class('section')
        section_frame.get_style_context().add_class('bottom-section')
        
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        section_frame.add(vbox)
        
        # Hardware usage bars
        self.usage_bars = {}
        
        for metric in ['CPU', 'RAM', 'GPU', 'Storage']:
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            hbox.get_style_context().add_class('usage-bar')
            
            label = Gtk.Label(label=f"{metric}:")
            label.get_style_context().add_class('usage-label')
            label.set_size_request(60, -1)
            hbox.pack_start(label, False, False, 0)
            
            progress = Gtk.ProgressBar()
            progress.get_style_context().add_class('progress-bar')
            progress.set_show_text(True)
            hbox.pack_start(progress, True, True, 0)
            
            self.usage_bars[metric] = progress
            vbox.pack_start(hbox, False, False, 0)
        
        parent.pack_start(section_frame, False, False, 0)
    
    def start_hardware_monitoring(self):
        def update_usage():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    GLib.idle_add(self.update_progress_bar, 'CPU', cpu_percent)
                    
                    # RAM usage
                    ram = psutil.virtual_memory()
                    ram_percent = ram.percent
                    GLib.idle_add(self.update_progress_bar, 'RAM', ram_percent)
                    
                    # GPU usage (if available)
                    try:
                        gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                                  capture_output=True, text=True, timeout=2)
                        if gpu_result.returncode == 0:
                            gpu_percent = float(gpu_result.stdout.strip())
                            GLib.idle_add(self.update_progress_bar, 'GPU', gpu_percent)
                        else:
                            GLib.idle_add(self.update_progress_bar, 'GPU', 0)
                    except:
                        GLib.idle_add(self.update_progress_bar, 'GPU', 0)
                    
                    # Storage usage
                    disk = psutil.disk_usage('/')
                    storage_percent = disk.percent
                    GLib.idle_add(self.update_progress_bar, 'Storage', storage_percent)
                    
                    time.sleep(2)
                except Exception as e:
                    print(f"Error updating hardware stats: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=update_usage, daemon=True)
        thread.start()
    
    def update_progress_bar(self, metric, value):
        if metric in self.usage_bars:
            progress_bar = self.usage_bars[metric]
            progress_bar.set_fraction(value / 100.0)
            progress_bar.set_text(f"{value:.1f}%")
    
    def launch_app(self, widget, command):
        try:
            subprocess.Popen(command, shell=True)
            self.hide_menu()
        except Exception as e:
            print(f"Error launching app: {e}")
    
    def launch_command(self, widget, command):
        try:
            subprocess.Popen(command, shell=True)
            self.hide_menu()
        except Exception as e:
            print(f"Error launching command: {e}")
    
    def show_menu(self):
        if self.window:
            # Position the window
            self.position_window()
            self.window.show_all()
            self.window.present()
            
            # For Wayland, we need to grab focus after showing
            GLib.timeout_add(100, self.grab_focus_delayed)
    
    def grab_focus_delayed(self):
        if self.window:
            self.window.grab_focus()
        return False
    
    def hide_menu(self):
        if self.window:
            self.window.hide()
    
    def position_window(self):
        # Get screen dimensions
        screen = Gdk.Screen.get_default()
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        
        # Get window size
        window_width, window_height = self.window.get_size()
        
        # Position in top right corner with some margin
        x = screen_width - window_width - 20
        y = 50
        
        self.window.move(x, y)
    
    def on_focus_out(self, widget, event):
        # Delay hiding to prevent immediate closure
        GLib.timeout_add(100, self.hide_menu_delayed)
        return False
    
    def hide_menu_delayed(self):
        self.hide_menu()
        return False
    
    def on_key_press(self, widget, event):
        if event.keyval == Gdk.KEY_Escape:
            self.hide_menu()
        return False
    
    def toggle_menu(self):
        if self.window and self.window.get_visible():
            self.hide_menu()
        else:
            self.show_menu()

def main():
    import sys
    
    # Initialize GTK
    Gtk.init(sys.argv)
    
    menu = GruvboxMenu()
    
    if len(sys.argv) > 1 and sys.argv[1] == "toggle":
        menu.toggle_menu()
    else:
        menu.show_menu()
    
    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\nExiting...")
        Gtk.main_quit()

if __name__ == "__main__":
    main()
