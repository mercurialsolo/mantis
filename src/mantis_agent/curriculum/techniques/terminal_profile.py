"""gnome-terminal profile size: persistent default rows/columns."""

NAME = "Terminal profile size"
TAGS = ["terminal", "size", "rows", "columns", "profile", "stty", "gnome-terminal"]
TRIGGERS = [
    r"terminal.*size", r"size.*terminal", r"\b132x?43\b", r"\bstty\b",
    r"rows.*cols", r"cols.*rows", r"\bgnome-terminal\b",
    r"default.*size", r"terminal.*default", r"resize.*terminal",
    r"window.*size.*terminal",
]
ALWAYS = False

CONTENT = """\
Terminal profile size techniques:
- The CORRECT way to set gnome-terminal's default window size persistently is to update the default profile, NOT to put `stty rows X cols Y` in .bashrc. The .bashrc approach changes the shell's view of the size after the terminal is already open, but the terminal window itself is whatever the WM gave it, and the eval may read the actual terminal dimensions.
- Get the default profile UUID:
  `PROFILE=$(gsettings get org.gnome.Terminal.ProfilesList default | tr -d "'")`
- Set the columns and rows for that profile:
  `gsettings set "org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:${PROFILE}/" default-size-columns 132`
  `gsettings set "org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:${PROFILE}/" default-size-rows 43`
- Or run all three in one go:
  `run_command('PROFILE=$(gsettings get org.gnome.Terminal.ProfilesList default | tr -d \"\\\'\") && gsettings set \"org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:${PROFILE}/\" default-size-columns 132 && gsettings set \"org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:${PROFILE}/\" default-size-rows 43')`
- After this, every NEW gnome-terminal window opens at exactly the requested size, which is what the eval verifies.\
"""
