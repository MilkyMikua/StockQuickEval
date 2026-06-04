# .\venv\Scripts\streamlit.exe run d:/fastapi-project/dashboard.py


import streamlit as st
import pandas as pd
from engines import (
    extract_fundamental_profile, 
    extract_single_stock_profile, 
    fetch_options_leverage_flow,
    extract_sentiment_ensemble
)

# Hardcode your authorized API Key securely
# https://aistudio.google.com/u/0/prompts/new_chat get your own API key
GEMINI_KEY = "AQ.Ab8R......"

st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("📊 Multi-Phase Quantitative Investment Engine")

# --- USER INPUT SIDEBAR ---
st.sidebar.header("Execution Controls")
ticker_input = st.sidebar.text_input("Enter Tickers (comma-separated)", value="TSLA, NVDA, CRWD")

if st.sidebar.button("Run Comprehensive Screening"):
    target_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    if not target_tickers:
        st.error("Please enter at least one valid ticker symbol.")
    else:
        with st.spinner("Processing multi-phase quant engine matrices..."):
            
            # Gather Phase 1 (Fundamental) Data
            fundamentals_list = []
            for t in target_tickers:
                f_profile = extract_fundamental_profile(t)
                if f_profile:
                    fundamentals_list.append(f_profile)
                    
            # Gather Phase 2 (Technical/Tactical) Data
            profiles_list = []
            for t in target_tickers:
                p_profile = extract_single_stock_profile(t)
                if p_profile:
                    profiles_list.append(p_profile)
                    
            # Gather Phase 3 (Sentiment Ensemble) Data
            sentiment_list = []
            for t in target_tickers:
                s_profile = extract_sentiment_ensemble(t, api_key=GEMINI_KEY)
                if s_profile:
                    sentiment_list.append(s_profile)

        # ========================================================
        # INITIALIZE THE COMPARTMENTALIZED UI TABS
        # ========================================================
        tab1, tab2, tab3 = st.tabs([
            "🧱 Phase 1: Fundamental Health", 
            "⏱️ Phase 2: Tactical & Stability", 
            "🧠 Phase 3: Sentiment Ensemble"
        ])

        # --------------------------------------------------------
        # TAB 1: PHASE 1 FUNDAMENTAL HEALTH SCREENER
        # --------------------------------------------------------
        with tab1:
            st.subheader("Quantitative Fundamental Moat Evaluation")
            
            # Simple Expandable Header to show summarized badge alerts cleanly
            with st.expander("🔍 Quick Inspection Health Badges", expanded=True):
                cols = st.columns(len(fundamentals_list))
                for idx, data in enumerate(fundamentals_list):
                    with cols[idx]:
                        score = data['fundamental_score']
                        st.metric(label=f"{data['ticker']} Score", value=f"{score}/100")
                        if score >= 80: st.success(data['rating'])
                        elif 60 <= score < 80: st.info(data['rating'])
                        else: st.warning(data['rating'])
            
            st.markdown("---")
            st.subheader("Raw Valuation & Balance Sheet Matrix")
            # Convert list of dictionaries to a clean dataframe for tabular display
            fund_df = pd.DataFrame(fundamentals_list).set_index('ticker')
            
            # Format numbers cleanly so they look polished on screen
            display_fund_df = fund_df.copy()
            
            display_fund_df['Company Name'] = display_fund_df['company_name']
            display_fund_df['Trailing P/E'] = display_fund_df['pe_trailing'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            display_fund_df['PEG Ratio'] = display_fund_df['peg_ratio'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            display_fund_df['Op. Margin (EBITDA)'] = display_fund_df['ebitda_margin'].map(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
            display_fund_df['Return on Equity'] = display_fund_df['roe'].map(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
            display_fund_df['YoY Rev Growth'] = display_fund_df['rev_growth'].map(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
            display_fund_df['Cash-to-Debt'] = display_fund_df['cash_to_debt'].map(lambda x: f"{x:.2f}x" if pd.notnull(x) else "N/A")
            display_fund_df['Current Ratio'] = display_fund_df['current_ratio'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
            # Format Free Cash Flow into readable Billions or Millions safely
            def format_fcf(val):
                if pd.isnull(val) or val is None: return "N/A"
                if abs(val) >= 1e9: return f"${val/1e9:.2f}B"
                return f"${val/1e6:.2f}M"
            display_fund_df['Free Cash Flow (TTM)'] = display_fund_df['free_cash_flow'].map(format_fcf)
            
            # Filter and order columns for final presentation
            columns_to_show = [
                'Company Name', 'fundamental_score', 'Trailing P/E', 'PEG Ratio', 
                'Op. Margin (EBITDA)', 'Return on Equity', 'YoY Rev Growth', 
                'Free Cash Flow (TTM)', 'Cash-to-Debt', 'Current Ratio'
            ]
            st.dataframe(display_fund_df[columns_to_show].rename(columns={'fundamental_score': 'Engine Score'}), use_container_width=True)

        # --------------------------------------------------------
        # TAB 2: PHASE 2 TACTICAL & STABILITY MONITOR
        # --------------------------------------------------------
        with tab2:
            st.subheader("⏱️ Real-Time Timing & Stability Verification")
            st.markdown("### 🎛️ Local Risk Threshold Controls")
            risk_mode = st.radio("Select Execution Risk Mode:", ["Conservative (Downside Protection)", "Aggressive (Momentum Capture)"], horizontal=True)
            
            max_rsi_allowed = 65.0 if "Conservative" in risk_mode else 78.0
            max_sfi_allowed = 0.70 if "Conservative" in risk_mode else 0.85
            
            st.markdown("---")
            st.subheader("📋 Tactical Timing Verdicts")
            flow_df = fetch_options_leverage_flow(target_tickers)
            
            for profile in profiles_list:
                t = profile['ticker']
                rsi = profile['rsi_14']
                obv_div = profile['obv_bullish_divergence']
                ratio = float(flow_df.loc[t, 'Raw_Ratio']) if not flow_df.empty and t in flow_df.index else profile.get('leverage_ratio', 0.0)

                st.markdown(f"#### **{t} Tactical Assessment**")
                if rsi > max_rsi_allowed and ratio > max_sfi_allowed:
                    st.error(f"🔴 **HIGH RISK WINDOW:** {t} is overcrowded (RSI: {rsi:.2f}) and experiencing heavy speculation ({ratio:.2f}x). **Do not buy.**")
                elif rsi < 45 and ratio <= max_sfi_allowed and obv_div:
                    st.success(f"🟢 **IDEAL ENTRY WINDOW:** {t} is consolidating quietly (RSI: {rsi:.2f}) with active institutional accumulation. **Safe Window.**")
                elif rsi < 45 and ratio <= max_sfi_allowed:
                    st.info(f"🔵 **HEALTHY CONSOLIDATION:** {t} is pulling back smoothly without derivatives panic. **Favorable entry building.**")
                else:
                    st.warning(f"🟡 **NEUTRAL CONDITIONS:** {t} is trading within standard technical boundaries (RSI: {rsi:.2f}, Options: {ratio:.2f}x).")
            
            st.markdown("---")
            st.subheader("📊 Underlying Technical & Volatility Matrix")
            if profiles_list and not flow_df.empty:
                tech_df = pd.DataFrame(profiles_list).set_index('ticker')
                master_tech = tech_df.join(flow_df, how='inner')
                display_tech_df = master_tech.copy()
                display_tech_df['Current Price'] = display_tech_df['current_price'].map('${:,.2f}'.format)
                display_tech_df['RSI (14d)'] = display_tech_df['rsi_14'].map('{:.2f}'.format)
                display_tech_df['Speculative Flow Ratio'] = display_tech_df['Raw_Ratio'].map('{:.2f}x'.format)
                display_tech_df['Institutional Accumulation'] = display_tech_df['obv_bullish_divergence'].map(lambda x: "🟢 Active" if x else "🔴 Standard")
                tech_cols = ['Current Price', 'RSI (14d)', 'Speculative Flow Ratio', 'SFI_Rating', 'Institutional Accumulation']
                st.dataframe(display_tech_df[tech_cols].rename(columns={'SFI_Rating': 'Derivatives Risk Status'}), use_container_width=True)

        # --------------------------------------------------------
        # TAB 3: PHASE 3 SENTIMENT ENSEMBLE LAYERS
        # --------------------------------------------------------
        with tab3:
            st.subheader("🧠 Institutional Sentiment & AI Narrative Analysis")
            
            # Render individual AI Narrative Analysis Panels
            for s_data in sentiment_list:
                t = s_data['ticker']
                fomo = s_data['ai_fomo_score']
                summary = s_data['ai_narrative_summary']
                
                st.markdown(f"#### **{t} Market Psychology Report**")
                
                # Visual Alert Boxes depending on the AI computed FOMO Score
                if fomo >= 75:
                    st.error(f"🔥 **EXTREME FOMO HYPE (Score: {fomo}/100):** {summary}")
                elif 40 <= fomo < 75:
                    st.info(f"⚖️ **STABLE NARRATIVE FLOW (Score: {fomo}/100):** {summary}")
                else:
                    st.success(f"❄️ **APATHY / CAPITAL ABANDONMENT (Score: {fomo}/100):** {summary}")
                    
            st.markdown("---")
            st.subheader("🏛️ Wall Street Macro Consensus Matrix")
            
            if sentiment_list:
                sent_df = pd.DataFrame(sentiment_list).set_index('ticker')
                display_sent_df = sent_df.copy()
                
                # Format numeric columns for presentation
                display_sent_df['Current Price'] = display_sent_df['current_price'].map('${:,.2f}'.format)
                display_sent_df['Analyst Target Mean'] = display_sent_df['target_mean'].map(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else "N/A")
                display_sent_df['Distance to Target'] = display_sent_df['analyst_distance_pct'].map('{:.2f}%'.format)
                display_sent_df['Wall St. Recommendation'] = display_sent_df['consensus_rating']
                display_sent_df['AI FOMO Index'] = display_sent_df['ai_fomo_score'].map('{} / 100'.format)
                
                sent_cols = ['Current Price', 'Analyst Target Mean', 'Distance to Target', 'Wall St. Recommendation', 'AI FOMO Index']
                st.dataframe(display_sent_df[sent_cols], use_container_width=True)