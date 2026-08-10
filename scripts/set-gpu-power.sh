#!/bin/bash
# Set GPU power limit to 225W
# Discrete GPU PCI address: 0000:03:00.0

GPU_PCI="0000:03:00.0"
LIMIT_WATTS="225"
LIMIT_MICROWATTS=$((LIMIT_WATTS * 1000000))

echo "Waking up GPU $GPU_PCI..."
echo "on" > "/sys/bus/pci/devices/$GPU_PCI/power/control"
sleep 1

HWMON_PATH=$(echo /sys/bus/pci/devices/$GPU_PCI/hwmon/hwmon*/power1_cap)
if [ -f "$HWMON_PATH" ]; then
    echo "Setting power limit to ${LIMIT_WATTS}W..."
    echo "$LIMIT_MICROWATTS" > "$HWMON_PATH"
    echo "Successfully set cap to: $(cat $HWMON_PATH) microwatts"
else
    echo "Error: Could not find power1_cap for GPU $GPU_PCI"
fi

echo "Allowing GPU to auto-suspend when idle..."
echo "auto" > "/sys/bus/pci/devices/$GPU_PCI/power/control"
