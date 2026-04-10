"""GNOME settings via gsettings command-line tool."""

NAME = "gsettings"
# Note: avoid generic words like "desktop" which collide with the ~/Desktop
# directory in file-management tasks. Use schema-specific terms instead.
TAGS = ["gsettings", "gnome", "schema", "preference", "dconf", "favorite"]
TRIGGERS = [
    r"gsettings", r"favorite", r"dim screen", r"idle.?delay",
    r"text scaling", r"magnif", r"notification", r"dconf",
    r"\bdesktop\b.*setting", r"\bgnome\b", r"\bdo not disturb\b",
    r"screensaver", r"lock.?screen",
]
ALWAYS = False

CONTENT = """\
gsettings techniques:
- Get a value: `run_command("gsettings get org.gnome.desktop.interface text-scaling-factor")`
- Set a scalar value: `run_command("gsettings set org.gnome.desktop.session idle-delay 0")`
- Set an array value (uses nested quotes — type_text/run_command handle the escaping):
  `run_command("gsettings set org.gnome.shell favorite-apps \\"['firefox.desktop', 'thunderbird.desktop']\\"")`
- If gsettings reports "No such schema", you may need DBUS context:
  `run_command("export DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus' && gsettings ...")`
- After setting, verify with `gsettings get` and you should see the new value.\
"""
