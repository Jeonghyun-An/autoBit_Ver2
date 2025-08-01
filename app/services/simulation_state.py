# app/services/simulation_state.py
simulation_state = {
    "krw": 1_000_000,
    "btc": 0.0,
    "history": [],  # 거래 기록
}

def reset_simulation():
    simulation_state["krw"] = 1_000_000
    simulation_state["btc"] = 0.0
    simulation_state["history"] = []

def get_simulation_status():
    return {
        "krw": round(simulation_state["krw"], 2),
        "btc": round(simulation_state["btc"], 8),
        "total": round(simulation_state["krw"] + simulation_state["btc"] * get_latest_price(), 2),
        "history": simulation_state["history"]
    }

def get_latest_price():
    from app.services.data_loader import fetch_upbit_data
    df = fetch_upbit_data()
    if df is None or df.empty:
        return 0.0
    return df["close"].iloc[-1]

def simulate_trade(action, price, model, strategy):
    if action == "BUY" and simulation_state["krw"] > 0:
        krw = simulation_state["krw"]
        btc_bought = krw / price
        simulation_state["btc"] += btc_bought
        simulation_state["krw"] = 0.0
        simulation_state["history"].append({
            "action": "BUY", "price": price, "btc": btc_bought, "model": model, "strategy": strategy
        })

    elif action == "SELL" and simulation_state["btc"] > 0:
        btc = simulation_state["btc"]
        krw_gained = btc * price
        simulation_state["krw"] += krw_gained
        simulation_state["btc"] = 0.0
        simulation_state["history"].append({
            "action": "SELL", "price": price, "krw": krw_gained, "model": model, "strategy": strategy
        })
