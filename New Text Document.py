import time
import math
import machine

# ── MOTOR PINS ───────────────
IN1 = machine.Pin(26, machine.Pin.OUT)
IN2 = machine.Pin(27, machine.Pin.OUT)
ENA = machine.PWM(machine.Pin(25))
ENA.freq(5000)

LED_GREEN = machine.Pin(2, machine.Pin.OUT)
LED_RED = machine.Pin(12, machine.Pin.OUT)

# ── NN WEIGHTS ───────────────
W1 = [1.700399, 1.333396, 0.198753, 1.078154, -0.206143, 0.150719, 0.546919, 0.190231, -0.234737, 0.271280, -0.231709, -0.232865, 0.120981, -0.956640, -0.862459, -0.281144, -0.506416, 0.157124, -0.454012, -0.706152, 0.732824, -0.112888, 0.033764, -0.712374, -0.564901, -0.166743, -0.396846, 0.268681, -0.072060, -0.012199, -0.141629, 1.209376, -0.006749, -0.528855, 0.411272, -0.610422, 0.104432, -0.979835, -0.664093, 0.098431, 0.897332, 0.576564, 0.060256, -0.039468, -0.579137, -0.071589, -0.120841, 0.627560, -0.691049, -1.587395, 0.472729, -0.041788, 0.185717, 0.539633, 0.885050, 1.139237, -0.419609, -0.154606, 0.165632, 0.487773, -0.239587, -0.092829, -0.553167, -0.598103, 1.226605, 1.476734, -0.028467, 0.720951, 0.211506, -0.055967, 0.146842, 0.798770, -0.051796, 0.744454, -1.312806, 0.385219, 0.018941, -0.174455, 0.023097, -1.019840, -0.178172, 0.105389, 0.561266, -0.311014, -0.524016, -0.390939, 0.318360, -0.004651, -0.264880, 0.256634, 0.048539, 0.484322, -0.351027, -0.163831, -0.196054, -0.731757] 
b1 = [-0.549175, 0.010000, 0.010000, 0.579866, 0.010000, 0.053069, 1.321923, 0.010000, -0.117145, -0.028924, -0.214202, 0.010000]

W2 = [0.087174, 0.106575, 0.002088, -0.095588, -0.577823, -0.172144, -0.097485, -0.327528, -0.118658, 0.164848, 0.772820, 0.071271, 0.008423, -0.030392, -0.783335, -0.086211, 0.024589, 0.974058, -0.208751, 0.123106, -0.179332, -0.477077, 0.424440, 0.306975, 0.343583, -0.371256, 0.572688, -0.633730, 0.239583, 0.916107, -0.448504, -0.231190, -0.024878, -0.205684, -0.639833, 0.027991, -1.145735, 0.193343, -0.375353, 1.183376, -0.319762, -0.375769, 2.311490, -0.502498, -0.416394, 0.527458, -0.631490, 0.075376, 0.093946, 0.319178, -0.504983, -0.539084, 0.213082, 0.121243, 0.076487, 0.141437, -0.290803, 0.094817, 0.110767, -0.291633, 2.099868, 0.193441, -0.486348, 0.034068, -0.397912, 0.857075, -0.489939, -0.335042, 1.784340, 0.176434, 0.283539, 0.774362, -0.286553, -0.307712, -0.363143, -0.078146, -0.031477, 0.068689, 0.737700, 0.337696, -0.078599, 0.588102, -0.092882, 1.110504, 0.232973, -0.349933, -0.437190, 0.164885, -0.091228, 0.284391, 0.026877, -0.029732, -0.404093, -0.618434, -0.188689, 0.349623]
b2 = [0.027893, -0.163702, -0.097989, 1.030319, -0.019835, -0.174446, 0.358984, -0.111668]

W3 = [0.119372, -0.623129, 0.070991, 1.764987, -0.440559, -2.449275, 0.391871, -0.567962, 0.187379, 0.281022, 0.529936, 1.247281, -0.689239, -0.667136, 0.399052, 0.255533, 0.242310, 1.913981, 0.213295, 0.569754, 0.476493, 0.367321, -0.116010, 0.378373, -0.373572, -0.136107, -0.433368, -0.793579, 1.157165, 0.900938, 0.197419, -0.806869, -0.254403, 0.574188, 0.322165, -1.999036, -0.357945, 1.187964, -0.765397, 0.107689]
b3 = [-0.348002, 0.000419, -0.122632, 0.696096, -0.175881]

CLASS_NAMES = ["SAFE","MODERATE","DANGER","HIGH_DANGER","EXTREME"]

# ── MOTOR CONTROL ────────────
def motor_stop():
    IN1.off()
    IN2.off()
    ENA.duty(0)

def motor_deploy():
    IN1.on()
    IN2.off()
    ENA.duty(800)

def motor_retract():
    IN1.off()
    IN2.on()
    ENA.duty(600)

# ── ACTIVATIONS ──────────────
def relu(x):
    return max(0, x)

def softmax(x):
    m = max(x)
    exps = [math.exp(i - m) for i in x]
    s = sum(exps)
    return [i/s for i in exps]

# ── NEURAL NETWORK ───────────
def nn_forward(inp):
    h1 = []
    for i in range(12):
        s = b1[i]
        for j in range(8):
            s += W1[i*8+j]*inp[j]
        h1.append(relu(s))

    h2 = []
    for i in range(8):
        s = b2[i]
        for j in range(12):
            s += W2[i*12+j]*h1[j]
        h2.append(relu(s))

    out = []
    for i in range(5):
        s = b3[i]
        for j in range(8):
            s += W3[i*8+j]*h2[j]
        out.append(s)

    probs = softmax(out)
    cls = probs.index(max(probs))
    return cls, probs[cls]

# ── FEATURES ────────────────
def build_features(speed, direction):
    return [
        speed/100,
        speed/120,
        direction/360,
        0.2,
        0.5,
        0.7,
        0.3,
        0.8
    ]

# ── STATE CONTROL (IMPORTANT FIX) ───────────
shield = False
danger_counter = 0
safe_counter = 0

def decision_logic(cls, conf, speed):
    global shield, danger_counter, safe_counter

    # 🔴 EXTREME overrides everything
    if cls == 4 and conf > 0.6:
        danger_counter += 2
    elif cls >= 2:
        danger_counter += 1
        safe_counter = 0
    else:
        safe_counter += 1
        danger_counter = 0

    # ── DEPLOY RULE ──
    if danger_counter >= 2 or speed >= 60:
        if not shield:
            return "DEPLOY"
        return "HOLD"

    # ── RETRACT RULE ──
    if safe_counter >= 3 and speed < 20:
        if shield:
            return "RETRACT"
        return "HOLD"

    return "HOLD"

# ── CSV READER ───────────────
def read_csv():
    with open("/wind_sensor_data.csv") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            yield parts[0], float(parts[1]), float(parts[2])

# ── MAIN LOOP ───────────────
while True:

    for ts, speed, direction in read_csv():

        print("\nTime:", ts)
        print("Speed:", speed, "| Direction:", direction)

        features = build_features(speed, direction)
        cls, conf = nn_forward(features)

        print("AI:", CLASS_NAMES[cls], "| Conf:", round(conf*100,1), "%")

        action = decision_logic(cls, conf, speed)

        # ── EXECUTE ACTION ──
        if action == "DEPLOY":
            print(">>> DEPLOY SHIELD")
            motor_deploy()
            LED_RED.on()
            LED_GREEN.off()
            shield = True

        elif action == "RETRACT":
            print(">>> RETRACT SHIELD")
            motor_retract()
            LED_GREEN.on()
            LED_RED.off()
            shield = False

        else:
            motor_stop()

        time.sleep(1)

    print("\n--- CSV LOOP RESTART ---\n")