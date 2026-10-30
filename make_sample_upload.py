"""
Generate sample_upload.csv — 300 rows with one hidden 8-account ring.

The ring shares device, IP and payment, posts an identical hype phrase
within 90 seconds, and uses accounts all under 10 hours old. Everything
else is a spread of genuine reviews with unique infrastructure.

Run:  py -3.14 make_sample_upload.py
"""
import csv
import random

random.seed(42)

HEADER = ["text_", "rating", "seller_id", "account_id", "account_age_hours",
          "device_fingerprint", "ip_block", "payment_instrument", "timestamp"]

GENUINE = [
    "The product arrived on time and matches the description.",
    "Nice headphones, sound quality is decent for the price.",
    "Screen was cracked on arrival. Returned and got a refund.",
    "Comfortable fit, but the cable is too short.",
    "Works fine. Battery lasts about two days per charge.",
    "Cheaper than expected but works okay for light use.",
    "Good value overall. Would buy again from this store.",
    "Packaging was damaged. Product itself seems fine.",
    "Delivery was slow but the item is genuine.",
    "Exactly as pictured. No complaints so far.",
    "Broke after two weeks. Cheap materials.",
    "Great for the price. Not premium but functional.",
    "Solid build quality. Fits my setup well.",
    "Instructions were confusing but after setup it works.",
]

TAIL = ["Order %d." % random.randint(1000, 9999),
        "%d days in." % random.randint(1, 30),
        "Would recommend.", "Shipping was quick."]

RING_PHRASE = ("UNBELIEVABLE 10/10 MUST BUY!!! Absolute game changer, "
               "best purchase ever.")
RING_SELLER = "SELLER-0042"
RING_DEVICE = "DEV-UPLOAD-RING-A7F3"
RING_IP = "198.51.100.0/24"
RING_PAYMENT = "PAY-UPLOAD-RING-4829"

rows = []
for i in range(292):
    rows.append({
        "text_": random.choice(GENUINE) + " " + random.choice(TAIL),
        "rating": str(random.randint(2, 5)),
        "seller_id": "SELLER-%04d" % random.randint(1, 40),
        "account_id": "ACCT-U%04d" % i,
        "account_age_hours": str(random.randint(100, 5000)),
        "device_fingerprint": "DEV-U%04d" % i,
        "ip_block": "203.0.%d.%d/24" % (random.randint(0, 255),
                                        random.randint(1, 254)),
        "payment_instrument": "PAY-U%04d" % i,
        "timestamp": str(1700000000 + i * 3600),
    })

base_ts = 1701000000
for i in range(8):
    rows.append({
        "text_": RING_PHRASE,
        "rating": "5",
        "seller_id": RING_SELLER,
        "account_id": "ACCT-RING-U%03d" % i,
        "account_age_hours": str(2.0 + i * 0.3),
        "device_fingerprint": RING_DEVICE,
        "ip_block": RING_IP,
        "payment_instrument": RING_PAYMENT,
        "timestamp": str(base_ts + i * 12),
    })

random.shuffle(rows)

with open("sample_upload.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=HEADER)
    w.writeheader()
    w.writerows(rows)

print("sample_upload.csv written: %d rows, ring of 8 at seller %s"
      % (len(rows), RING_SELLER))