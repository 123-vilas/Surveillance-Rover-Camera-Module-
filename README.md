# Surveillance Rover Camera Module

## Overview
The **Surveillance Rover Camera Module** is an integral part of a remotely controlled rover that streams live video. This module is responsible for capturing and transmitting real-time video feed, enhancing surveillance and monitoring capabilities.

## Features
- **Live Video Streaming**: Captures and transmits real-time video from the rover’s camera.
- **ESP32-WROVER Integration**: Uses the ESP32-WROVER module for efficient video processing.
- **Low Latency Communication**: Optimized for minimal delay in video transmission.
- **Android App Compatibility**: Designed to work seamlessly with the Android control application.
- **Lightweight & Power Efficient**: Ensures optimal power consumption for extended surveillance.

## Hardware Requirements
- **ESP32-WROVER Module**
- **Camera Module (e.g., OV2640)**
- **Battery Power Supply**
- **Rover Chassis with Motor Drivers**
- **Wi-Fi or Bluetooth Connectivity Module (if needed)**

## Software Requirements
- **Arduino IDE / PlatformIO**
- **ESP32 Camera Library**
- **WebSocket or HTTP Server for Video Transmission**
- **Android App for Control & Display**

## Installation & Setup
1. **Flash Firmware**:
   - Install the ESP32 Camera library in Arduino IDE.
   - Flash the provided firmware to the ESP32-WROVER.
2. **Network Configuration**:
   - Set up Wi-Fi credentials in the code.
   - Ensure the Android app and ESP32 module are on the same network.
3. **Android App Setup**:
   - Install the Surveillance Rover control app.
   - Connect to the ESP32 stream via WebSocket/HTTP.
4. **Testing the Camera**:
   - Power on the rover and access the video stream.
   - Verify real-time feed on the Android app.

## Usage
- Start the Android control app.
- Connect to the rover’s camera module.
- Control the rover and view live video feed.
- Adjust camera settings for better clarity.

## Troubleshooting
- **No Video Feed?** Check the camera connections and ESP32 firmware.
- **High Latency?** Optimize the Wi-Fi network and reduce resolution if needed.
- **App Not Connecting?** Verify IP settings and ensure the server is running.

## Future Enhancements
- Improve video quality and stability.
- Implement object detection for better surveillance.
- Add cloud storage for video recordings.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Contributors
- **[VILAS N  AND SUDARSHAN T S]** - Developer
- **Contributors Welcome!** Feel free to submit issues and pull requests.

## Contact
For questions or contributions, reach out via [GitHub Issues](https://github.com/123-vilas/Surveillance-Rover-Camera-Module-/issues).

