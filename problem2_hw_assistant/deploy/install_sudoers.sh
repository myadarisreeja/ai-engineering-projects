#!/usr/bin/env bash
# Installs the narrow sudoers rule that lets the assistant read DMI memory data (dmidecode -t memory).
# Review deploy/hw-assistant.sudoers.template first. Run with: sudo ./deploy/install_sudoers.sh [username]
set -euo pipefail
TARGET_USER="${1:-${SUDO_USER:-}}"
[ -n "$TARGET_USER" ] || { echo "usage: sudo $0 <username>"; exit 1; }
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
DMI="$(command -v dmidecode || true)"
[ -x "$DMI" ] || { echo "dmidecode not installed (apt install dmidecode / dnf install dmidecode)"; exit 1; }
DMI="$(readlink -f "$DMI")"
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp)"
sed -e "s|__USER__|$TARGET_USER|" -e "s|__DMIDECODE__|$DMI|" "$HERE/hw-assistant.sudoers.template" > "$TMP"
visudo -cf "$TMP" >/dev/null || { echo "generated sudoers failed validation"; rm -f "$TMP"; exit 1; }
install -m 0440 -o root -g root "$TMP" /etc/sudoers.d/hw-assistant
rm -f "$TMP"
echo "Installed /etc/sudoers.d/hw-assistant for $TARGET_USER:"; cat /etc/sudoers.d/hw-assistant
echo "Remove any time with: sudo rm /etc/sudoers.d/hw-assistant"
