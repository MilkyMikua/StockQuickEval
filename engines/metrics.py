import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
from google import genai
import json


warnings.filterwarnings('ignore')

# ==========================================
# CORE DATA ACQUISITION
# ==========================================

def fetch_sector_data(tickers: list, period: str = "2y") -> pd.DataFrame:
    """Fetches historical daily close prices for a list of tickers."""
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, period=period, progress=False)
    
    if len(tickers) == 1:
        closes = pd.DataFrame(data['Close']).rename(columns={'Close': tickers[0]})
    else:
        closes = data['Close']
        
    return closes.dropna(how='all')


# ==========================================
# RAW METRIC EXTRACTIONS (INDEPENDENT)
# ==========================================

def calculate_technical_metrics(closes_df: pd.DataFrame) -> tuple:
    """Calculates historical RSI and 50-day SMA distance matrix."""
    if closes_df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    rsis = closes_df.apply(lambda x: ta.rsi(x, length=14))
    sma50 = closes_df.apply(lambda x: ta.sma(x, length=50))
    dist_sma = (closes_df - sma50) / sma50
    return rsis, dist_sma


def get_historical_percentile(series: pd.Series) -> float:
    """Calculates the historical percentile of the current value in a series."""
    valid_series = series.dropna()
    if valid_series.empty:
        return np.nan
    latest_val = valid_series.iloc[-1]
    return float((valid_series < latest_val).mean() * 100)


def fetch_options_leverage_flow(tickers: list) -> pd.DataFrame:
    """Calculates options leverage ratio relative to underlying equity volume."""
    flow_data = []
    for ticker_symbol in tickers:
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="5d")
            if hist.empty:
                continue
                
            latest_stock_vol = hist['Volume'].iloc[-1]
            if latest_stock_vol == 0 or pd.isna(latest_stock_vol):
                continue

            exp_dates = ticker.options
            if not exp_dates:
                continue

            total_opt_vol = 0
            for date in exp_dates[:2]:
                chain = ticker.option_chain(date)
                calls_vol = chain.calls['volume'].sum() if 'volume' in chain.calls else 0
                puts_vol = chain.puts['volume'].sum() if 'volume' in chain.puts else 0
                total_opt_vol += (np.nan_to_num(calls_vol) + np.nan_to_num(puts_vol))

            leverage_ratio = (total_opt_vol * 100) / latest_stock_vol
            flow_data.append({
                'Ticker': ticker_symbol,
                'Stock_Volume': int(latest_stock_vol),
                'Option_Volume_Shares': int(total_opt_vol * 100),
                'Opt_to_Stock_Ratio': float(leverage_ratio)
            })
        except Exception:
            continue

    if not flow_data:
        return pd.DataFrame()
    return pd.DataFrame(flow_data).set_index('Ticker')


def extract_single_stock_profile(ticker_symbol: str) -> dict:
    """Extracts raw profile metrics for a single stock, including specialized OBV."""
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="6mo")

    if hist.empty:
        return {}

    closes = hist['Close']
    latest_stock_vol = hist['Volume'].iloc[-1]

    # Technical Indicators
    rsi_series = ta.rsi(closes, length=14)
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else np.nan

    # Vectorized On-Balance Volume (OBV)
    hist['OBV'] = np.where(hist['Close'] > hist['Close'].shift(1), hist['Volume'],
                  np.where(hist['Close'] < hist['Close'].shift(1), -hist['Volume'], 0)).cumsum()

    # Move Trends
    hist['Close_SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['OBV_SMA_20'] = hist['OBV'].rolling(window=20).mean()

    obv_bullish_divergence = bool((hist['Close'].iloc[-1] < hist['Close_SMA_20'].iloc[-1]) and 
                                  (hist['OBV'].iloc[-1] > hist['OBV_SMA_20'].iloc[-1]))

    # Options Calculations
    exp_dates = ticker.options
    total_opt_vol = 0
    if exp_dates:
        for date in exp_dates[:2]:
            chain = ticker.option_chain(date)
            calls_vol = chain.calls['volume'].sum() if 'volume' in chain.calls else 0
            puts_vol = chain.puts['volume'].sum() if 'volume' in chain.puts else 0
            total_opt_vol += (np.nan_to_num(calls_vol) + np.nan_to_num(puts_vol))

    leverage_ratio = float((total_opt_vol * 100) / latest_stock_vol if latest_stock_vol > 0 else 0)

    return {
        "ticker": ticker_symbol,
        "current_price": float(closes.iloc[-1]),
        "rsi_14": rsi,
        "leverage_ratio": leverage_ratio,
        "obv_bullish_divergence": obv_bullish_divergence
    }

def fetch_options_leverage_flow(tickers: list) -> pd.DataFrame:
    """
    Calculates options leverage ratio and transforms it into a 
    Standardized Speculative Flow Index (SFI).
    """
    flow_data = []
    for ticker_symbol in tickers:
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="5d")
            if hist.empty:
                continue
                
            latest_stock_vol = hist['Volume'].iloc[-1]
            if latest_stock_vol == 0 or pd.isna(latest_stock_vol):
                continue

            exp_dates = ticker.options
            if not exp_dates:
                continue

            total_opt_vol = 0
            for date in exp_dates[:2]:
                chain = ticker.option_chain(date)
                calls_vol = chain.calls['volume'].sum() if 'volume' in chain.calls else 0
                puts_vol = chain.puts['volume'].sum() if 'volume' in chain.puts else 0
                total_opt_vol += (np.nan_to_num(calls_vol) + np.nan_to_num(puts_vol))

            leverage_ratio = (total_opt_vol * 100) / latest_stock_vol
            raw_ratio = float(leverage_ratio)

            # --- TRANSFORMATION INTO SPECULATIVE FLOW INDEX (SFI) ---
            # Bound the score between 0 and 100 for standard indexing layouts
            sfi_score = min(100.0, max(0.0, raw_ratio * 100)) 
            
            if raw_ratio >= 0.8:
                sfi_rating = "CRITICAL"
                emotion = "Extreme Speculation / Gamma Squeeze Risk"
            elif 0.6 <= raw_ratio < 0.8:
                sfi_rating = "ELEVATED"
                emotion = "High Speculative Flow / Retail Momentum"
            elif 0.4 <= raw_ratio < 0.6:
                sfi_rating = "NORMAL"
                emotion = "Standard Equity Driven Volatility"
            else:
                sfi_rating = "CONSERVATIVE"
                emotion = "Low Derivatives Activity / Institutional Flow"

            flow_data.append({
                'Ticker': ticker_symbol,
                'Stock_Volume': int(latest_stock_vol),
                'Option_Volume_Shares': int(total_opt_vol * 100),
                'Raw_Ratio': round(raw_ratio, 4),
                'SFI_Score': round(sfi_score, 2),
                'SFI_Rating': sfi_rating,
                'Market_Emotion': emotion
            })
        except Exception:
            continue

    if not flow_data:
        return pd.DataFrame()
    return pd.DataFrame(flow_data).set_index('Ticker')    


def extract_fundamental_profile(ticker_symbol: str) -> dict:
    """
    Fetches comprehensive corporate financials and executes an advanced
    4-Pillar Quantitative Grading Matrix scored out of 100.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Guardrail: Handle ETFs cleanly without breaking fundamental math matrices
        quote_type = info.get('quoteType', '').upper()
        if 'ETF' in quote_type:
            return {
                "ticker": ticker_symbol, "is_etf": True,
                "company_name": info.get('shortName', 'Unknown ETF'),
                "fundamental_score": 50, "rating": "ETF MODE",
                "pe_trailing": None, "pe_forward": None, "peg_ratio": None,
                "profit_margin": None, "ebitda_margin": None, "roe": None,
                "rev_growth": None, "free_cash_flow": None,
                "cash_to_debt": None, "current_ratio": None
            }
            
        # 1. EXTRACT BALANCED METRICS WITH SAFE FALLBACK VALUES
        company_name = info.get('longName', ticker_symbol)
        pe_trailing = info.get('trailingPE')
        pe_forward = info.get('forwardPE')
        peg_ratio = info.get('pegRatio')               # Valuation check vs Growth Rate
        
        profit_margin = info.get('profitMargins')       # Net Margin
        ebitda_margin = info.get('ebitdaMargins')       # Raw Operational Moat Margin
        roe = info.get('returnOnEquity')                 # Management Capital Compounding
        rev_growth = info.get('revenueGrowth')           # YoY Top-line velocity
        free_cash_flow = info.get('freeCashflow')       # Raw Cash Flow (Actual structural survival asset)
        
        total_cash = info.get('totalCash', 0)
        total_debt = info.get('totalDebt', 0)
        current_ratio = info.get('currentRatio')       # Short-term liquidity runway check
        
        # Calculate Cash-to-Debt Ratio safely
        if total_debt and total_debt > 0:
            cash_to_debt = total_cash / total_debt
        else:
            cash_to_debt = 2.0 if total_cash and total_cash > 0 else 1.0

        # ========================================================
        # 4-PILLAR MATHEMATICAL SCORING ALGORITHM (MAX 100 PTS)
        # ========================================================
        score = 0
        
        # ---- PILLAR 1: MOAT & PROFITABILITY (MAX 25 POINTS) ----
        # Gross Margin surrogate (defaulting safely to Net Margin if missing)
        gross_margin = info.get('grossMargins', profit_margin or 0)
        if gross_margin and gross_margin >= 0.40: score += 10
        elif gross_margin and gross_margin >= 0.20: score += 5
        
        if ebitda_margin and ebitda_margin >= 0.20: score += 15
        elif ebitda_margin and ebitda_margin >= 0.10: score += 10
        elif ebitda_margin and ebitda_margin > 0: score += 5

        # ---- PILLAR 2: RUNWAY & LIQUIDITY (MAX 25 POINTS) ----
        if cash_to_debt >= 1.5: score += 15
        elif 1.0 <= cash_to_debt < 1.5: score += 10
        elif 0.5 <= cash_to_debt < 1.0: score += 5
        
        if current_ratio and current_ratio >= 1.5: score += 10
        elif current_ratio and 1.0 <= current_ratio < 1.5: score += 7
        elif current_ratio and current_ratio < 1.0: score += 2  # Liquidity pinch risk

        # ---- PILLAR 3: GROWTH & CASH FLOW TRUTH (MAX 25 POINTS) ----
        if rev_growth and rev_growth >= 0.20: score += 15
        elif rev_growth and rev_growth >= 0.08: score += 10
        elif rev_growth and rev_growth > 0: score += 5
        elif rev_growth and rev_growth <= 0: score -= 5  # Demote shrinking operations
        
        # Grade Cold Cash Flow reality
        if free_cash_flow and free_cash_flow > 0: score += 10
        elif free_cash_flow and free_cash_flow <= 0: score -= 5 # Massive warning penalty for cash bleed

        # ---- PILLAR 4: CAPITAL EFFICIENCY & PRICING (MAX 25 POINTS) ----
        # Low PEG means you aren't overpaying for growth momentum
        if peg_ratio and 0.0 < peg_ratio <= 1.2: score += 15
        elif peg_ratio and 1.2 < peg_ratio <= 2.2: score += 10
        elif peg_ratio and peg_ratio > 2.2: score += 2   # Growth value trap
        else: score += 8                                 # Fair baseline if metric fails
        
        if roe and roe >= 0.20: score += 10
        elif roe and roe >= 0.10: score += 6
        elif roe and roe > 0: score += 2

        # Final Bounding limits
        score = max(0, min(100, score))
        
        # Structural qualitative brackets
        if score >= 85: rating = "Strong Institutional Moat"
        elif 65 <= score < 85: rating = "Healthy / High Quality"
        elif 45 <= score < 65: rating = "Speculative / Asset Light"
        else: rating = "High Fundamental Risk Floor"
        
        return {
            "ticker": ticker_symbol, "is_etf": False, "company_name": company_name,
            "fundamental_score": score, "rating": rating,
            "pe_trailing": pe_trailing, "pe_forward": pe_forward, "peg_ratio": peg_ratio,
            "profit_margin": profit_margin, "ebitda_margin": ebitda_margin, "roe": roe,
            "rev_growth": rev_growth, "free_cash_flow": free_cash_flow,
            "cash_to_debt": cash_to_debt, "current_ratio": current_ratio
        }
        
    except Exception as e:
        print(f"Error compiling fundamentals for {ticker_symbol}: {e}")
        return {"ticker": ticker_symbol, "fundamental_score": 0, "rating": "Error Processing Data", "is_etf": False}

def extract_sentiment_ensemble(ticker_symbol: str, api_key: str) -> dict:
    """
    Fetches Wall Street consensus targets and processes news headlines 
    using the Gemini API to construct a crowd psychology FOMO score.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # 1. Wall Street Analyst Consensus Data
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 1.0
        target_mean = info.get('targetMeanPrice')
        
        if target_mean and current_price:
            analyst_distance_pct = ((target_mean - current_price) / current_price) * 100
        else:
            analyst_distance_pct = 0.0
            
        consensus_rating = info.get('recommendationKey', 'N/A').replace('_', ' ').title()
        
        # 2. Fetch News Headlines for the AI Pipeline
        news_items = ticker.news[:8]  # Snag the top 8 recent headlines
        headlines = [item.get('title', '') for item in news_items if item.get('title')]
        
        # Default fallback values if no news is found
        ai_fomo_score = 50
        ai_narrative_summary = "Insufficient recent headline flow to establish crowd sentiment tracking parameters."
        
        # 3. Trigger the Cloud Gemini AI Engine
        if headlines and api_key:
            # Initialize the clean, modern GenAI client
            client = genai.Client(api_key=api_key)
            
            # Construct a strict prompt requiring a structural JSON response
            news_block = "\n".join([f"- {h}" for h in headlines])
            prompt = f"""
            Analyze the following stock market news headlines for the ticker {ticker_symbol}:
            {news_block}
            
            Evaluate the current retail/institutional crowd psychology and determine:
            1. An overall FOMO score from 0 to 100 (where 0 is absolute panic/abandonment, 50 is cold neutrality, and 100 is extreme euphoric FOMO/hype bubble chasing).
            2. A single concise, high-level structural summary sentence assessing the narrative direction.
            
            You must return your output strictly in valid JSON format with exactly two keys: "fomo_score" (integer) and "summary" (string). Do not add markdown code blocks or wrapper text.
            """
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            # Clean up potential markdown formatting blocks the LLM might append
            raw_text = response.text.strip().replace("```json", "").replace("```", "")
            
            try:
                ai_data = json.loads(raw_text)
                ai_fomo_score = int(ai_data.get('fomo_score', 50))
                ai_narrative_summary = ai_data.get('summary', ai_narrative_summary)
            except Exception:
                # If JSON parsing hits an parsing edge-case, do a quick safe fallback
                if "fomo_score" in raw_text:
                    ai_narrative_summary = "AI processed text successfully, parsing fallback active."
        
        return {
            "ticker": ticker_symbol,
            "current_price": current_price,
            "target_mean": target_mean if target_mean else "N/A",
            "analyst_distance_pct": analyst_distance_pct,
            "consensus_rating": consensus_rating,
            "ai_fomo_score": ai_fomo_score,
            "ai_narrative_summary": ai_narrative_summary
        }
        
    except Exception as e:
        print(f"Error compiling sentiment for {ticker_symbol}: {e}")
        return {
            "ticker": ticker_symbol,
            "current_price": 0.0,
            "target_mean": "N/A",
            "analyst_distance_pct": 0.0,
            "consensus_rating": "Data Error",
            "ai_fomo_score": 50,
            "ai_narrative_summary": f"Failed to hook up sentiment pipes: {e}"
        }