import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from uuid import uuid4

# =====================================================
# CONFIG
# =====================================================

N_ROWS = 10000

TODAY = pd.Timestamp.today().normalize()
CURRENT_YEAR = TODAY.year

# =====================================================
# LOOKUPS
# =====================================================

MAKE_MODEL_ENGINE = {
    "Freightliner": {
        "Cascadia": ["Detroit DD15", "Cummins X15"],
        "Coronado": ["Detroit DD13"],
        "Columbia": ["Detroit Series 60"]
    },
    "Peterbilt": {
        "579": ["Cummins X15"],
        "389": ["Cummins X15"],
        "567": ["PACCAR MX13"]
    },
    "Kenworth": {
        "T680": ["PACCAR MX13", "Cummins X15"],
        "W900": ["Cummins X15"],
        "T880": ["PACCAR MX13"]
    },
    "Volvo": {
        "VNL760": ["Volvo D13"],
        "VNL860": ["Volvo D13"]
    },
    "International": {
        "LT625": ["A26 Engine"],
        "ProStar": ["MaxxForce 13"]
    },
    "Mack": {
        "Anthem": ["MP8"]
    }
}

MAKE_WEIGHTS = {
    "Freightliner": 0.35,
    "Peterbilt": 0.20,
    "Kenworth": 0.20,
    "Volvo": 0.15,
    "International": 0.07,
    "Mack": 0.03
}

LOCATIONS = [
    "Dallas, TX",
    "Houston, TX",
    "Fort Worth, TX",
    "Atlanta, GA",
    "Chicago, IL",
    "Phoenix, AZ",
    "Memphis, TN",
    "Kansas City, MO",
    "Indianapolis, IN",
    "Salt Lake City, UT"
]

TRUCK_TYPES = {
    "Sleeper": 0.60,
    "Day Cab": 0.25,
    "Dump": 0.05,
    "Box Truck": 0.05,
    "Flatbed": 0.05
}

SLEEPER_SIZES = ['48"', '60"', '72"', '80"', '86"']

AXLES = ["6x4", "6x2", "4x2"]
AXLE_WEIGHTS = [0.75, 0.15, 0.10]

TRANSMISSION_TYPES = ["Automatic", "Manual"]
TRANS_WEIGHTS = [0.70, 0.30]

SOURCES = {
    "truckpaper.com": "retail",
    "commercialtrucktrader.com": "retail",
    "ritchiebros.com": "auction",
    "ironplanet.com": "auction",
    "equipmentfacts.com": "auction"
}

SOURCE_WEIGHTS = [0.35, 0.30, 0.15, 0.10, 0.10]

VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"

# =====================================================
# HELPERS
# =====================================================

def weighted_choice(d):
    return np.random.choice(
        list(d.keys()),
        p=list(d.values())
    )

def generate_vin():
    return ''.join(random.choices(VIN_CHARS, k=17))

def generate_year():
    years = (
        list(range(2000, 2006)),
        list(range(2006, 2011)),
        list(range(2011, 2016)),
        list(range(2016, 2021)),
        list(range(2021, 2026))
    )

    bucket = np.random.choice(
        [0,1,2,3,4],
        p=[0.05,0.15,0.30,0.35,0.15]
    )

    return random.choice(years[bucket])

def generate_mileage(year):
    age = max(1, CURRENT_YEAR - year)

    annual = np.random.randint(
        60000,
        120000
    )

    mileage = annual * age

    noise = np.random.normal(
        0,
        mileage * 0.15
    )

    mileage += noise

    return max(50000, int(mileage))

def generate_price(
    year,
    mileage,
    truck_type,
    apu,
    engine
):
    age = CURRENT_YEAR - year

    base = 170000

    price = (
        base
        - age * 6000
        - mileage * 0.04
    )

    if truck_type == "Sleeper":
        price += 8000

    if apu:
        price += 3000

    if "Cummins" in engine:
        price += 2500

    if "DD15" in engine:
        price += 4000

    price += np.random.normal(
        0,
        5000
    )

    return round(
        np.clip(price, 5000, 180000),
        2
    )

def generate_listing_url(source):
    listing_id = random.randint(
        10000000,
        99999999
    )

    return (
        f"https://www.{source}/listing/"
        f"{listing_id}"
    )

# =====================================================
# DATA GENERATION
# =====================================================

rows = []

for idx in range(N_ROWS):

    source = np.random.choice(
        list(SOURCES.keys()),
        p=SOURCE_WEIGHTS
    )

    source_type = SOURCES[source]

    year = generate_year()

    make = np.random.choice(
        list(MAKE_WEIGHTS.keys()),
        p=list(MAKE_WEIGHTS.values())
    )

    model = random.choice(
        list(MAKE_MODEL_ENGINE[make].keys())
    )

    engine = random.choice(
        MAKE_MODEL_ENGINE[make][model]
    )

    vin = generate_vin()

    truck_type = weighted_choice(TRUCK_TYPES)

    sleeper_size = (
        random.choice(SLEEPER_SIZES)
        if truck_type == "Sleeper"
        else None
    )

    mileage = generate_mileage(year)

    apu = (
        np.random.rand() < 0.50
        if truck_type == "Sleeper"
        else np.random.rand() < 0.10
    )

    overhaul_paperwork = (
        mileage > 900000
        and np.random.rand() < 0.40
    )

    transmission_type = np.random.choice(
        TRANSMISSION_TYPES,
        p=TRANS_WEIGHTS
    )

    if transmission_type == "Automatic":
        transmission_speed = random.choice(
            [10, 12]
        )
    else:
        transmission_speed = random.choice(
            [10, 13, 18]
        )

    wheelbase_inches = int(
        np.clip(
            np.random.normal(230, 20),
            180,
            280
        )
    )

    axles = np.random.choice(
        AXLES,
        p=AXLE_WEIGHTS
    )

    location = random.choice(
        LOCATIONS
    )

    first_seen_date = (
        TODAY -
        timedelta(
            days=random.randint(0, 1200)
        )
    )

    days_on_market = int(
        np.random.lognormal(
            mean=4.3,
            sigma=1.0
        )
    )

    last_seen_date = (
        first_seen_date +
        timedelta(days=days_on_market)
    )

    if last_seen_date > TODAY:
        last_seen_date = TODAY

    if last_seen_date == TODAY:
        status = "active"
    else:
        status = np.random.choice(
            ["ended", "withdrawn"],
            p=[0.55, 0.45]
        )

    auction_date = None

    if source_type == "auction":
        delta = (
            last_seen_date -
            first_seen_date
        ).days

        auction_date = (
            first_seen_date +
            timedelta(
                days=random.randint(
                    0,
                    max(1, delta)
                )
            )
        )

    price = generate_price(
        year,
        mileage,
        truck_type,
        apu,
        engine
    )

    is_relisted = (
        np.random.rand() < 0.15
    )

    is_duplicate = (
        np.random.rand() < 0.05
    )

    review_flag = (
        mileage > 1500000
        or price < 5000
        or year > CURRENT_YEAR
    )

    created_at = (
        first_seen_date +
        timedelta(
            hours=random.randint(0, 23)
        )
    )

    updated_at = max(
        created_at,
        pd.Timestamp(last_seen_date)
    )

    rows.append({
        "source": source,
        "source_type": source_type,
        "year": year,
        "make": make,
        "model": model,
        "vin": vin,
        "price": price,
        "mileage": mileage,
        "location": location,
        "engine": engine,
        "axles": axles,
        "transmission_type": transmission_type,
        "transmission_speed": transmission_speed,
        "wheelbase_inches": wheelbase_inches,
        "apu": apu,
        "truck_type": truck_type,
        "sleeper_size": sleeper_size,
        "overhaul_paperwork": overhaul_paperwork,
        "listing_url": generate_listing_url(source),
        "auction_date": auction_date,
        "first_seen_date": first_seen_date,
        "last_seen_date": last_seen_date,
        "is_duplicate": is_duplicate,
        "review_flag": review_flag,
        "is_relisted": is_relisted,
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at
    })

df = pd.DataFrame(rows)

print(df.shape)
print(df.head())

df.to_csv(
    "truck_marketplace.csv",
    index=False
)