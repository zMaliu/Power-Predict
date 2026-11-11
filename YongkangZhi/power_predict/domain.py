"""Shared target and column contracts; no I/O or model dependencies."""
TARGETS = {"industry": "全行业用电", "resident": "城乡居民用电"}
COLS = {"date": "date", "industry": "industry_kwh", "resident": "resident_kwh",
        "tem": "tem", "tem_max": "tem_max", "tem_min": "tem_min", "rhum": "rhum",
        "precip": "precip", "wind_speed": "wind_speed", "weather_type": "weather_type"}
WEATHER_KEYS = ("tem", "tem_max", "tem_min", "rhum", "precip", "wind_speed", "weather_type")
