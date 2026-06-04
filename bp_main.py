#!/usr/bin/env python3
# ================================================
# BP Estimation System — Main Code
# File: /home/ritam2002/bp_project/bp_main.py
# ================================================
# Connections:
#   ESP32 USB      -> USB port
#   LCD VCC        -> RPi Pin 2  (5V)
#   LCD GND        -> RPi Pin 6  (GND)
#   LCD SDA        -> RPi Pin 3  (GPIO2)
#   LCD SCL        -> RPi Pin 5  (GPIO3)
#   BUTTON_START   -> RPi Pin 11 (GPIO17) + GND Pin 9   [handled by bp_launcher.py]
#   BUTTON_LIVE    -> RPi Pin 13 (GPIO27) + GND Pin 14  [handled here — jumps to Live]
# ================================================

import os
import time
import threading
import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ── LCD ──────────────────────────────────────────
try:
    from RPLCD.i2c import CharLCD
    LCD_AVAILABLE = True
except ImportError:
    LCD_AVAILABLE = False
    print("RPLCD not installed.  Run: pip install RPLCD smbus2")

# ── GPIO ─────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("RPi.GPIO not installed.  Run: pip install RPi.GPIO")

# ================================================
# CONFIG
# ================================================
LCD_I2C_ADDRESS   = 0x27
DATA_DIR          = "/home/ritam2002/bp_project"
SERIAL_BAUD       = 115200
SERIAL_TIMEOUT    = 20

ENCODER_SAVE_PATH = os.path.join(DATA_DIR, "saved_encoder.keras")
RF_SAVE_PATH      = os.path.join(DATA_DIR, "rf_model.joblib")
MASTER_CSV        = os.path.join(DATA_DIR, "master_dataset.csv")
WINDOWS_PER_READ  = 5

# ── GPIO pin (BCM numbering) ──────────────────────
PIN_LIVE = 27     # Button 2 → Pin 13 → jumps to Live Inference

# ── LCD result hold time ──────────────────────────
LCD_HOLD_SECONDS = 3

# ── Calibration offsets ───────────────────────────
# Run Option 5 to calculate these automatically.
# Then hardcode the printed values here.
SBP_OFFSET = -9    # mmHg
DBP_OFFSET = 12    # mmHg
# ================================================

encoder_model   = None
rf_model        = None
lcd             = None

btn_live_event  = threading.Event()   # set by GPIO ISR when Button 2 pressed
stop_live_flag  = threading.Event()   # set to exit live inference loop


# ================================================
# Sampling layer (needed for .keras save/load)
# ================================================
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def get_config(self):
        return super().get_config()


# ================================================
# GPIO SETUP — Button 2 only
# ================================================
def init_gpio():
    if not GPIO_AVAILABLE:
        return False
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(PIN_LIVE, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(PIN_LIVE, GPIO.FALLING,
                              callback=_btn_live_cb, bouncetime=300)
        print(f"GPIO ready — BUTTON_LIVE on GPIO{PIN_LIVE} (Pin 13)")
        return True
    except Exception as e:
        print(f"GPIO init failed: {e}")
        return False


def _btn_live_cb(channel):
    """ISR — fires when Button 2 (GPIO27) is pressed."""
    btn_live_event.set()
    stop_live_flag.set()    # also exits any running live loop cleanly


def cleanup_gpio():
    if GPIO_AVAILABLE:
        try:
            GPIO.cleanup()
        except Exception:
            pass


# ================================================
# LCD
# ================================================
def init_lcd():
    global lcd
    if not LCD_AVAILABLE:
        return False
    try:
        lcd = CharLCD(
            i2c_expander    = 'PCF8574',
            address         = LCD_I2C_ADDRESS,
            port            = 1,
            cols            = 16,
            rows            = 2,
            dotsize         = 8,
            charmap         = 'A02',
            auto_linebreaks = True
        )
        lcd_show("BP Estimator", "Booting up...")
        time.sleep(1)
        print("LCD initialised OK")
        return True
    except Exception as e:
        print(f"LCD init failed: {e}")
        print("  Check address with: sudo i2cdetect -y 1")
        return False


def lcd_show(line1="", line2=""):
    if lcd is None:
        return
    try:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string(str(line1)[:16].ljust(16))
        lcd.cursor_pos = (1, 0)
        lcd.write_string(str(line2)[:16].ljust(16))
    except Exception as e:
        print(f"LCD write error: {e}")


def lcd_show_result(line1="", line2="", hold=LCD_HOLD_SECONDS):
    """Show BP result and keep it on screen for `hold` seconds."""
    lcd_show(line1, line2)
    time.sleep(hold)


# ================================================
# SIGNAL VALIDATION — autocorrelation periodicity
# ================================================
def is_valid_ppg_signal(waveform, acf_threshold=0.45):
    wf = waveform[0].astype(float)
    n  = len(wf)

    wf  = wf - np.mean(wf)
    std = np.std(wf)

    if std < 1e-3:
        return False, "Flat signal — sensor disconnected?"

    wf_norm = wf / std
    acf     = np.correlate(wf_norm, wf_norm, mode='full')
    acf     = acf[n - 1:]
    acf     = acf / acf[0]

    min_lag = 8
    max_lag = n // 2
    if max_lag <= min_lag:
        return False, "Window too short for ACF check"

    peak = float(np.max(acf[min_lag:max_lag]))
    if peak < acf_threshold:
        return False, (
            f"Non-periodic (ACF={peak:.2f} < {acf_threshold:.2f})"
            " — no finger / poor contact"
        )
    return True, f"Valid (ACF={peak:.2f})"


# ================================================
# MODEL SAVE / LOAD
# ================================================
def save_models():
    global encoder_model, rf_model
    if encoder_model is None or rf_model is None:
        print("Nothing to save — train first.")
        return False

    os.makedirs(DATA_DIR, exist_ok=True)
    ok = True
    try:
        encoder_model.save(ENCODER_SAVE_PATH)
        print(f"  Encoder saved  -> {ENCODER_SAVE_PATH}")
    except Exception as e:
        print(f"  Encoder save FAILED: {e}")
        ok = False
    try:
        joblib.dump(rf_model, RF_SAVE_PATH)
        print(f"  RF model saved -> {RF_SAVE_PATH}")
    except Exception as e:
        print(f"  RF save FAILED: {e}")
        ok = False
    return ok


def load_models():
    global encoder_model, rf_model
    if not (os.path.exists(ENCODER_SAVE_PATH) and os.path.exists(RF_SAVE_PATH)):
        return False
    try:
        print("  Loading encoder...", end=' ', flush=True)
        encoder_model = tf.keras.models.load_model(
            ENCODER_SAVE_PATH,
            custom_objects = {'Sampling': Sampling},
            compile        = False
        )
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        encoder_model = None
        return False
    try:
        print("  Loading RF model...", end=' ', flush=True)
        rf_model = joblib.load(RF_SAVE_PATH)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        rf_model = None
        return False
    return True


# ================================================
# SERIAL
# ================================================
def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if any(chip in p.description.upper()
               for chip in ['CH340', 'CP210', 'CP2102', 'UART', 'USB']):
            print(f"ESP32 found: {p.device}  ({p.description})")
            return p.device
    print("\nAvailable ports:")
    for p in ports:
        print(f"  {p.device}  -  {p.description}")
    return input("\nEnter port manually (e.g. /dev/ttyUSB0): ").strip()


def open_serial_and_wait(port, boot_timeout=15):
    print(f"Opening {port} at {SERIAL_BAUD} baud...")
    ser = serial.Serial(port, SERIAL_BAUD, timeout=2)
    print("Waiting for ESP32 to boot", end='', flush=True)
    ser.reset_input_buffer()
    deadline = time.time() + boot_timeout

    while time.time() < deadline:
        time.sleep(0.1)
        print('.', end='', flush=True)
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                continue
            if line:
                print(f"\n  ESP32: {line}")
            if 'READY' in line.upper():
                print("ESP32 is ready!")
                return ser

    print("\nESP32_READY not received — proceeding anyway...")
    ser.reset_input_buffer()
    return ser


# ================================================
# READ ONE WAVEFORM PACKET
# ================================================
def read_waveform(ser):
    data     = []
    deadline = time.time() + SERIAL_TIMEOUT

    while time.time() < deadline:
        if stop_live_flag.is_set():
            return None
        if ser.in_waiting == 0:
            time.sleep(0.005)
            continue
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        try:
            val = float(line.split()[0])
            data.append(val)
        except:
            continue
        if len(data) >= 101:
            return np.array(data[:101]).reshape(1, -1)

    print("Timeout: no waveform received from ESP32.")
    return None


# ================================================
# DATA LOADING
# ================================================
def load_data(file_path=None):
    if file_path is None:
        all_files = sorted([
            f for f in os.listdir(DATA_DIR)
            if f.endswith('.csv') and os.path.isfile(os.path.join(DATA_DIR, f))
        ])
        if not all_files:
            print(f"No CSV files found in {DATA_DIR}")
            return None, None

        print(f"\nCSV files in {DATA_DIR}:")
        for i, f in enumerate(all_files):
            print(f"  [{i}] {f}")

        while True:
            choice = input("\nEnter number to select: ").strip()
            if choice.isdigit() and 0 <= int(choice) < len(all_files):
                file_path = os.path.join(DATA_DIR, all_files[int(choice)])
                break
            print(f"  Please enter 0 to {len(all_files) - 1}")

    file_path = file_path.strip().strip('"').strip("'")
    if os.path.isdir(file_path):
        print(f"That is a folder: {file_path}")
        return None, None
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None

    print(f"\nLoading: {file_path}")
    data = pd.read_csv(file_path)
    print(f"  Shape: {data.shape}  |  Waveform columns: {data.shape[1] - 3}")
    X = data.iloc[:, 3:].values
    y = data.iloc[:, 0:2].values
    return X, y


# ================================================
# NORMALIZATION
# ================================================
def normalize(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std  = np.std(X,  axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


# ================================================
# PLOT
# ================================================
def plot_waveforms(X, title, save_path=None):
    plt.figure(figsize=(10, 5))
    for i in range(min(5, len(X))):
        plt.plot(X[i], label=f"Sample {i+1}")
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("ADC Amplitude")
    plt.legend()
    plt.tight_layout()
    out = save_path or os.path.join(DATA_DIR, "last_plot.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Plot saved -> {out}")


# ================================================
# BUILD VAE
# ================================================
def build_vae(input_dim, latent_dim=8):
    inp = layers.Input(shape=(input_dim,))
    x   = layers.Dense(128, activation='relu')(inp)
    x   = layers.Dense(64,  activation='relu')(x)
    zm  = layers.Dense(latent_dim, name='z_mean')(x)
    zlv = layers.Dense(latent_dim, name='z_log_var')(x)
    z   = Sampling(name='z')([zm, zlv])
    encoder = models.Model(inp, [zm, zlv, z], name='encoder')

    li  = layers.Input(shape=(latent_dim,))
    x   = layers.Dense(64,  activation='relu')(li)
    x   = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(input_dim)(x)
    decoder = models.Model(li, out, name='decoder')

    class VAE(tf.keras.Model):
        def __init__(self, enc, dec):
            super().__init__()
            self.encoder = enc
            self.decoder = dec

        def train_step(self, data):
            if isinstance(data, tuple):
                data = data[0]
            with tf.GradientTape() as tape:
                zm, zlv, z = self.encoder(data)
                rec  = self.decoder(z)
                rl   = tf.reduce_mean(tf.reduce_sum(tf.square(data - rec), axis=1))
                kl   = -0.5 * tf.reduce_mean(
                           tf.reduce_sum(1 + zlv - tf.square(zm) - tf.exp(zlv), axis=1))
                loss = rl + kl
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {"loss": loss, "recon_loss": rl, "kl_loss": kl}

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    return vae, encoder


# ================================================
# LIVE INFERENCE — called by menu option 3 AND button 2
# ================================================
def run_live_inference():
    if encoder_model is None or rf_model is None:
        print("Train the model first (Option 1)!")
        lcd_show("Train first!", "")
        return

    port = find_esp32_port()
    if not port:
        print("No ESP32 port found.")
        return

    lcd_show("Connecting...", "ESP32 USB")
    try:
        ser = open_serial_and_wait(port)
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        lcd_show("Serial Error!", "Check USB")
        return

    ser.write(b"START_ACQUISITION\n")
    time.sleep(0.5)
    ser.reset_input_buffer()

    stop_live_flag.clear()
    btn_live_event.clear()

    lcd_show("Place Finger", "On PPG Sensor")
    print("Place your finger on the PPG sensor.")
    print("Press Ctrl+C or BUTTON_LIVE (GPIO27) to stop.")
    print(f"[Offsets] SBP={SBP_OFFSET:+.0f} mmHg  DBP={DBP_OFFSET:+.0f} mmHg")
    print(f"[LCD]     Result held for {LCD_HOLD_SECONDS}s per reading")
    print("─" * 60)

    reading_count = 0
    failed_count  = 0

    try:
        while not stop_live_flag.is_set():
            waveform = read_waveform(ser)

            if stop_live_flag.is_set():
                break

            if waveform is None:
                print("No signal — resending START...")
                lcd_show("No Signal!", "Retrying...")
                try:
                    ser.write(b"START_ACQUISITION\n")
                    time.sleep(0.5)
                    ser.reset_input_buffer()
                except serial.SerialException:
                    print("Serial connection lost.")
                    break
                continue

            is_valid, reason = is_valid_ppg_signal(waveform)

            if not is_valid:
                failed_count += 1
                print(f"[REJECT {failed_count:02d}]  {reason}")
                lcd_show("No Finger?", "Adjust sensor")
                if failed_count >= 8:
                    print("\n  ⚠ Persistent rejects. Check:")
                    print("    • Finger firmly on TCRT1000 red LED")
                    print("    • Sensor clean (no dust/moisture)")
                    print("    • Shield from bright ambient light")
                    print("    • No dark nail polish")
                    lcd_show("Check sensor!", "See terminal")
                    failed_count = 0
                continue

            # Valid — predict + calibration offsets
            failed_count  = 0
            Wn            = normalize(waveform)
            z_mean, _, _  = encoder_model.predict(Wn, verbose=0)
            pred          = rf_model.predict(z_mean)[0]
            sbp           = float(pred[0]) + SBP_OFFSET
            dbp           = float(pred[1]) + DBP_OFFSET
            reading_count += 1

            if   sbp < 120 and dbp < 80:  label = "Normal"
            elif sbp < 130 and dbp < 80:  label = "Elevated"
            elif sbp < 140 or  dbp < 90:  label = "High Stage1"
            else:                          label = "High Stage2"

            print(f"[{reading_count:03d}]  "
                  f"SBP: {sbp:5.1f} mmHg  |  "
                  f"DBP: {dbp:5.1f} mmHg  |  {label}")

            # Hold result on LCD for LCD_HOLD_SECONDS
            lcd_show_result(f"SBP:{sbp:.0f} DBP:{dbp:.0f}",
                            label[:16],
                            hold=LCD_HOLD_SECONDS)

    except KeyboardInterrupt:
        print("\nLive inference stopped.")

    finally:
        lcd_show("Stopped.", "Back to menu")
        try:
            ser.write(b"STOP_ACQUISITION\n")
            time.sleep(0.3)
            ser.close()
        except Exception:
            pass
        stop_live_flag.clear()
        btn_live_event.clear()
        print("Returned to menu.\n")


# ================================================
# CALIBRATION — Option 5
# ================================================
def run_calibration():
    global SBP_OFFSET, DBP_OFFSET

    if encoder_model is None or rf_model is None:
        print("Train the model first (Option 1)!")
        return

    port = find_esp32_port()
    if not port:
        print("No ESP32 port found.")
        return
    try:
        ser = open_serial_and_wait(port)
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return

    try:
        n = int(input("\nHow many paired readings? (recommended: 5): ").strip() or 5)
    except ValueError:
        n = 5

    sbp_errors = []
    dbp_errors = []

    print(f"\nFor each of {n} readings:")
    print("  1. Take a cuff reading.")
    print("  2. Place finger on sensor.")
    print("  3. Enter cuff values when prompted.\n")

    ser.write(b"START_ACQUISITION\n")
    time.sleep(0.5)
    ser.reset_input_buffer()
    stop_live_flag.clear()

    for i in range(n):
        print(f"── Reading {i+1}/{n} ──")
        try:
            cuff_sbp = float(input("  Cuff SBP (mmHg): ").strip())
            cuff_dbp = float(input("  Cuff DBP (mmHg): ").strip())
        except ValueError:
            print("  Invalid — skipping.")
            continue

        print("  Place finger on sensor now...")
        lcd_show(f"Calibrate {i+1}/{n}", "Place finger")

        waveform = None
        for _ in range(10):
            wf = read_waveform(ser)
            if wf is None:
                continue
            is_valid, reason = is_valid_ppg_signal(wf)
            if is_valid:
                waveform = wf
                break
            print(f"  Rejected ({reason}) — retrying...")

        if waveform is None:
            print("  No valid signal — skipping.")
            continue

        Wn           = normalize(waveform)
        z_mean, _, _ = encoder_model.predict(Wn, verbose=0)
        pred         = rf_model.predict(z_mean)[0]
        raw_sbp      = float(pred[0])
        raw_dbp      = float(pred[1])
        sbp_err      = cuff_sbp - raw_sbp
        dbp_err      = cuff_dbp - raw_dbp
        sbp_errors.append(sbp_err)
        dbp_errors.append(dbp_err)
        print(f"  Cuff:{cuff_sbp:.0f}/{cuff_dbp:.0f}  "
              f"Raw:{raw_sbp:.1f}/{raw_dbp:.1f}  "
              f"Err: SBP={sbp_err:+.1f} DBP={dbp_err:+.1f}")

    ser.write(b"STOP_ACQUISITION\n")
    time.sleep(0.3)
    ser.close()
    stop_live_flag.clear()

    if not sbp_errors:
        print("\nNo valid readings — calibration aborted.")
        return

    new_sbp = round(float(np.mean(sbp_errors)), 1)
    new_dbp = round(float(np.mean(dbp_errors)), 1)

    print(f"\n╔══════════════════════════════╗")
    print(f"║     CALIBRATION RESULT        ║")
    print(f"╠══════════════════════════════╣")
    print(f"║ Readings : {len(sbp_errors):<19}║")
    print(f"║ SBP offset : {new_sbp:+.1f} mmHg        ║")
    print(f"║ DBP offset : {new_dbp:+.1f} mmHg        ║")
    print(f"╠══════════════════════════════╣")
    print(f"║ Old: SBP={SBP_OFFSET:+.1f} DBP={DBP_OFFSET:+.1f}       ║")
    print(f"╚══════════════════════════════╝")

    if input("\nApply these offsets now? [Y/n]: ").strip().lower() != 'n':
        SBP_OFFSET = new_sbp
        DBP_OFFSET = new_dbp
        print(f"Applied: SBP={SBP_OFFSET:+.1f}  DBP={DBP_OFFSET:+.1f}")
        print(f"\n⚠ To make permanent, edit lines in bp_main.py CONFIG:")
        print(f"    SBP_OFFSET = {SBP_OFFSET}")
        print(f"    DBP_OFFSET = {DBP_OFFSET}")
        lcd_show("Calibrated!", f"SBP{SBP_OFFSET:+.0f} DBP{DBP_OFFSET:+.0f}")
    else:
        print("Offsets unchanged.")


# ================================================
# DATA COLLECTION
# ================================================
def collect_and_save():
    os.makedirs(DATA_DIR, exist_ok=True)
    session_csv = os.path.join(DATA_DIR,
                               f"session_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    port = find_esp32_port()
    if not port:
        print("No ESP32 port found.")
        return

    lcd_show("Data Collect", "Connecting...")
    try:
        ser = open_serial_and_wait(port)
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        lcd_show("Serial Error!", "Check USB")
        return

    print("\nPress Ctrl+C to stop and save.\n")
    lcd_show("Connect OK!", "Ready to read")
    time.sleep(1)
    stop_live_flag.clear()

    total_rows     = 0
    header_written = os.path.exists(MASTER_CSV)

    try:
        while True:
            print("─" * 50)
            print(f"Reading #{total_rows + 1}")
            try:
                sbp_ref = float(input("  Reference SBP (mmHg) [0 to skip]: ").strip() or 0)
                dbp_ref = float(input("  Reference DBP (mmHg) [0 to skip]: ").strip() or 0)
                n_win   = int(input(f"  Windows [{WINDOWS_PER_READ}]: ").strip()
                               or WINDOWS_PER_READ)
            except ValueError:
                print("  Invalid — using defaults.")
                sbp_ref, dbp_ref, n_win = 0, 0, WINDOWS_PER_READ

            print(f"\nCollecting {n_win} window(s)...")
            lcd_show("Place Finger", "On PPG Sensor")

            ser.write(b"START_ACQUISITION\n")
            time.sleep(0.5)
            ser.reset_input_buffer()

            collected = 0
            rows      = []
            num_s     = None

            while collected < n_win:
                wf = read_waveform(ser)
                if wf is None:
                    print("  No signal — retrying...")
                    lcd_show("No Signal!", "Retrying...")
                    continue
                collected += 1
                num_s      = wf.shape[1]
                rows.append(
                    [sbp_ref, dbp_ref, time.strftime("%Y-%m-%dT%H:%M:%S")]
                    + wf.flatten().tolist()
                )
                print(f"  Window {collected}/{n_win} OK")
                lcd_show(f"Got {collected}/{n_win}", f"{sbp_ref:.0f}/{dbp_ref:.0f}mmHg")

            ser.write(b"STOP_ACQUISITION\n")
            time.sleep(0.3)

            cols   = (["SBP", "DBP", "timestamp"] +
                      [f"s{i+1}" for i in range(num_s)])
            df_new = pd.DataFrame(rows, columns=cols)
            df_new.to_csv(session_csv, mode='a', index=False,
                          header=not os.path.exists(session_csv))
            df_new.to_csv(MASTER_CSV,  mode='a', index=False,
                          header=not header_written)
            header_written = True
            total_rows    += len(rows)

            print(f"\n  Saved {len(rows)} row(s).  Total: {total_rows}")
            lcd_show(f"Saved! #{total_rows}", "Next reading?")

            if input("\nCollect another? [Y/n]: ").strip().lower() == 'n':
                break

    except KeyboardInterrupt:
        print("\nData collection stopped.")
        lcd_show("Saved!", f"{total_rows} rows")

    finally:
        try:
            ser.write(b"STOP_ACQUISITION\n")
            time.sleep(0.3)
            ser.close()
        except Exception:
            pass
        stop_live_flag.clear()

    print(f"\nDone.  Total rows: {total_rows}")


# ================================================
# Non-blocking input — returns None if Button 2
# is pressed before the user finishes typing
# ================================================
def _input_with_button_check(prompt):
    result_holder = [None]
    input_done    = threading.Event()

    def _read():
        try:
            result_holder[0] = input(prompt).strip()
        except Exception:
            result_holder[0] = ""
        input_done.set()

    t = threading.Thread(target=_read, daemon=True)
    t.start()

    while not input_done.is_set():
        if btn_live_event.is_set():
            return None           # button pressed — interrupt input
        time.sleep(0.05)

    return result_holder[0]


# ================================================
# MAIN MENU
# ================================================
def main():
    global encoder_model, rf_model

    os.makedirs(DATA_DIR, exist_ok=True)
    init_lcd()
    init_gpio()

    print("\nChecking for saved models...")
    if load_models():
        print("Models loaded — ready!\n")
        lcd_show("Models Loaded!", "Ready")
    else:
        print("No saved models. Train first (Option 1).\n")
        lcd_show("BP Estimator", "Train first")

    while True:
        model_status = "Loaded  " if (encoder_model and rf_model) else "Not trained"
        print(f"\n╔══════════════════════════════════╗")
        print(f"║      BP ESTIMATION SYSTEM         ║")
        print(f"║  Model : {model_status:<24}║")
        print(f"║  SBP{SBP_OFFSET:+.0f}  DBP{DBP_OFFSET:+.0f} (offsets)      ║")
        print(f"╠══════════════════════════════════╣")
        print(f"║  1. Train & Save Model            ║")
        print(f"║  2. Test on CSV                   ║")
        print(f"║  3. Live Inference (ESP32)        ║")
        print(f"║  4. Collect & Save Data           ║")
        print(f"║  5. Calibrate Offsets             ║")
        print(f"║  6. Exit                          ║")
        print(f"╠══════════════════════════════════╣")
        print(f"║  BUTTON_LIVE (GPIO27, Pin 13)     ║")
        print(f"║  → instantly starts Live Infer.   ║")
        print(f"╚══════════════════════════════════╝")

        # Check if button was already pressed before we got here
        if btn_live_event.is_set():
            btn_live_event.clear()
            print("\n[BUTTON_LIVE] Jumping to Live Inference...")
            lcd_show("Button pressed", "Starting Live")
            time.sleep(0.3)
            run_live_inference()
            continue

        choice = _input_with_button_check("Choice: ")

        # Button pressed while waiting for keyboard input
        if choice is None:
            btn_live_event.clear()
            print("\n[BUTTON_LIVE] Jumping to Live Inference...")
            lcd_show("Button pressed", "Starting Live")
            time.sleep(0.3)
            run_live_inference()
            continue

        # ── Menu options ──────────────────────────
        if choice == "1":
            print("\n── TRAIN MODE ──")
            X_train, y_train = load_data()
            if X_train is None:
                continue

            Xn = normalize(X_train)
            plot_waveforms(Xn, "Training Waveforms — first 5",
                           os.path.join(DATA_DIR, "train_plot.png"))

            input_dim = X_train.shape[1]
            print(f"\nInput dimension: {input_dim}")
            print("Training VAE — 20 epochs...")
            lcd_show("Training VAE", "Please wait...")

            vae, encoder_model = build_vae(input_dim=input_dim)
            vae.fit(Xn, epochs=20, batch_size=64, verbose=1)

            print("\nExtracting latent features...")
            z_mean, _, _ = encoder_model.predict(Xn, verbose=0)

            print("Training Random Forest...")
            lcd_show("Training RF...", "Please wait...")
            rf_model = RandomForestRegressor(
                n_estimators = 200,
                max_depth    = 10,
                n_jobs       = -1,
                random_state = 42
            )
            rf_model.fit(z_mean, y_train)
            print("\nTraining complete!")
            lcd_show("Training Done!", "Saving...")
            if save_models():
                print("Models saved!")
                lcd_show("Saved!", "Ready :)")
            else:
                print("Save failed.")
                lcd_show("Save failed!", "Check disk")

        elif choice == "2":
            if encoder_model is None or rf_model is None:
                print("Train first!")
                continue

            print("\n── TEST MODE ──")
            X_test, y_test = load_data()
            if X_test is None:
                continue

            Xn           = normalize(X_test)
            plot_waveforms(Xn, "Test Waveforms — first 5",
                           os.path.join(DATA_DIR, "test_plot.png"))
            z_mean, _, _ = encoder_model.predict(Xn, verbose=0)
            y_pred       = rf_model.predict(z_mean)

            y_pred_cal       = y_pred.copy()
            y_pred_cal[:, 0] += SBP_OFFSET
            y_pred_cal[:, 1] += DBP_OFFSET

            mae_sbp = mean_absolute_error(y_test[:, 0], y_pred_cal[:, 0])
            mae_dbp = mean_absolute_error(y_test[:, 1], y_pred_cal[:, 1])
            sd_sbp  = np.std(y_test[:, 0] - y_pred_cal[:, 0])
            sd_dbp  = np.std(y_test[:, 1] - y_pred_cal[:, 1])

            print(f"\n╔══════════════════════════════╗")
            print(f"║     RESULTS (calibrated)      ║")
            print(f"╠══════════════════════════════╣")
            print(f"║ SBP MAE : {mae_sbp:6.2f} mmHg      ║")
            print(f"║ DBP MAE : {mae_dbp:6.2f} mmHg      ║")
            print(f"║ SBP SD  : {sd_sbp:6.2f} mmHg      ║")
            print(f"║ DBP SD  : {sd_dbp:6.2f} mmHg      ║")
            print(f"╠══════════════════════════════╣")
            for i in range(min(5, len(y_test))):
                a_s, a_d = y_test[i]
                p_s, p_d = y_pred_cal[i]
                print(f"║  [{i+1}] Act:{a_s:.0f}/{a_d:.0f}  Pred:{p_s:.0f}/{p_d:.0f}  ║")
            print(f"╚══════════════════════════════╝")
            lcd_show(f"SBP MAE:{mae_sbp:.1f}", f"DBP MAE:{mae_dbp:.1f}")

        elif choice == "3":
            run_live_inference()

        elif choice == "4":
            print("\n── DATA COLLECTION MODE ──")
            lcd_show("Data Collect", "Starting...")
            collect_and_save()

        elif choice == "5":
            print("\n── CALIBRATION MODE ──")
            lcd_show("Calibrating...", "Follow steps")
            run_calibration()

        elif choice == "6":
            print("\nShutting down...")
            lcd_show("Goodbye!", "System Off")
            time.sleep(1)
            if lcd:
                try:
                    lcd.clear()
                    lcd.close()
                except Exception:
                    pass
            cleanup_gpio()
            break

        else:
            print("Invalid — enter 1 to 6.")


# ── Entry point ───────────────────────────────────
if __name__ == "__main__":
    main()
