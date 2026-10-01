
from ib_async import *
import pandas as pd
import asyncio

def test_connection():
    util.startLoop()

    # Create IB instance
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)  # Paper trading
        print("Connected to IBKR API")
    except Exception as e:
        print(f"Connection failed: {e}")
    
    print(f"Is connected: {ib.isConnected()}")
