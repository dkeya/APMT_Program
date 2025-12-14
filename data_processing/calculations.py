# apmt_dashboard/data_processing/calculations.py

import pandas as pd
import numpy as np
import re
from utils.helpers import to_num, coalesce_first, yn


# ----------------------------------------------------------------------
# HERD METRICS
# ----------------------------------------------------------------------
def calculate_herd_metrics(processor):
    """
    Calculate herd composition and productivity metrics.

    Interpretation:
    - Rates are "events in the last 1 month per 100 head of small ruminants".
      Denominator is current herd size (approximation to animals-at-risk).
    """
    try:
        df = processor.df

        # Define all columns we need upfront
        herd_columns = [
            "total_sheep",
            "total_goats",
            "total_sr",
            "pct_female",
            "pct_male",
            "total_births",
            "total_mortality",
            "total_losses",
            "birth_rate_per_100",
            "mortality_rate_per_100",
            "loss_rate_per_100",
        ]

        # Initialize all columns with 0.0 if missing
        for col in herd_columns:
            if col not in df.columns:
                df[col] = 0.0

        # -----------------------------
        # Current herd size
        # -----------------------------
        sheep_cols = [
            c
            for c in [
                "C3. Number of Rams currently owned (total: at home + away + relatives/friends)",
                "C3. Number of Ewes currently owned (total: at home + away + relatives/friends)",
            ]
            if c in df.columns
        ]

        goat_cols = [
            c
            for c in [
                "C3. Number of Bucks currently owned (total: at home + away + relatives/friends)",
                "C3. Number of Does currently owned (total: at home + away + relatives/friends)",
            ]
            if c in df.columns
        ]

        all_animal_cols = sheep_cols + goat_cols
        if all_animal_cols:
            df[all_animal_cols] = df[all_animal_cols].apply(
                lambda x: to_num(x).fillna(0)
            )

        if sheep_cols:
            df["total_sheep"] = df[sheep_cols].sum(axis=1)
        if goat_cols:
            df["total_goats"] = df[goat_cols].sum(axis=1)

        df["total_sr"] = df["total_sheep"] + df["total_goats"]

        # -----------------------------
        # Gender composition
        # -----------------------------
        female_sheep_col = (
            "C3. Number of Ewes currently owned (total: at home + away + relatives/friends)"
        )
        female_goat_col = (
            "C3. Number of Does currently owned (total: at home + away + relatives/friends)"
        )
        male_sheep_col = (
            "C3. Number of Rams currently owned (total: at home + away + relatives/friends)"
        )
        male_goat_col = (
            "C3. Number of Bucks currently owned (total: at home + away + relatives/friends)"
        )

        idx = df.index

        if female_sheep_col in df.columns:
            female_sheep = to_num(df[female_sheep_col]).fillna(0)
        else:
            female_sheep = pd.Series(0.0, index=idx)

        if female_goat_col in df.columns:
            female_goats = to_num(df[female_goat_col]).fillna(0)
        else:
            female_goats = pd.Series(0.0, index=idx)

        if male_sheep_col in df.columns:
            male_sheep = to_num(df[male_sheep_col]).fillna(0)
        else:
            male_sheep = pd.Series(0.0, index=idx)

        if male_goat_col in df.columns:
            male_goats = to_num(df[male_goat_col]).fillna(0)
        else:
            male_goats = pd.Series(0.0, index=idx)

        total_female = female_sheep + female_goats
        total_male = male_sheep + male_goats

        valid = df["total_sr"] > 0

        with np.errstate(divide="ignore", invalid="ignore"):
            df.loc[valid, "pct_female"] = np.clip(
                total_female[valid] / df.loc[valid, "total_sr"] * 100, 0, 100
            )
            df.loc[valid, "pct_male"] = np.clip(
                total_male[valid] / df.loc[valid, "total_sr"] * 100, 0, 100
            )
            df.loc[~valid, ["pct_female", "pct_male"]] = 0.0

        # -----------------------------
        # Births, mortality, and losses (last 1 month)
        # -----------------------------
        def existing(cols):
            return [c for c in cols if c in df.columns]

        birth_cols = existing(
            [
                "C4. Number of Rams born in the last 1 month",
                "C4. Number of Ewes born in the last 1 month",
                "C4. Number of Bucks born in the last 1 month",
                "C4. Number of Does born in the last 1 month",
            ]
        )

        mort_cols = existing(
            [
                "C5. Number of Rams that died in the last 1 month",
                "C5. Number of Ewes that died in the last 1 month",
                "C5. Number of Bucks that died in the last 1 month",
                "C5. Number of Does that died in the last 1 month",
            ]
        )

        loss_cols = existing(
            [
                "C6. Number of Rams lost/not found or lost to wild animals in the last 1 month",
                "C6. Number of Ewes lost/not found or lost to wild animals in the last 1 month",
                "C6. Number of Bucks lost/not found or lost to wild animals in the last 1 month",
                "C6. Number of Does lost/not found or lost to wild animals in the last 1 month",
            ]
        )

        all_event_cols = birth_cols + mort_cols + loss_cols
        if all_event_cols:
            df[all_event_cols] = df[all_event_cols].apply(
                lambda x: to_num(x).fillna(0)
            )

        if birth_cols:
            df["total_births"] = df[birth_cols].sum(axis=1)
        if mort_cols:
            df["total_mortality"] = df[mort_cols].sum(axis=1)
        if loss_cols:
            df["total_losses"] = df[loss_cols].sum(axis=1)

        # -----------------------------
        # Monthly rates per 100 head
        # -----------------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            df["birth_rate_per_100"] = 0.0
            df["mortality_rate_per_100"] = 0.0
            df["loss_rate_per_100"] = 0.0

            valid_mask = df["total_sr"] > 0
            if valid_mask.any():
                denom = df.loc[valid_mask, "total_sr"]

                df.loc[valid_mask, "birth_rate_per_100"] = (
                    df.loc[valid_mask, "total_births"] / denom * 100
                ).fillna(0)
                df.loc[valid_mask, "mortality_rate_per_100"] = (
                    df.loc[valid_mask, "total_mortality"] / denom * 100
                ).fillna(0)
                df.loc[valid_mask, "loss_rate_per_100"] = (
                    df.loc[valid_mask, "total_losses"] / denom * 100
                ).fillna(0)

        processor.df = df

    except Exception:
        # Conservative fallback: ensure columns exist and are numeric
        df = processor.df
        for col in [
            "total_sheep",
            "total_goats",
            "total_sr",
            "pct_female",
            "pct_male",
            "total_births",
            "total_mortality",
            "total_losses",
            "birth_rate_per_100",
            "mortality_rate_per_100",
            "loss_rate_per_100",
        ]:
            if col not in df.columns:
                df[col] = 0.0
        processor.df = df


# ----------------------------------------------------------------------
# PROFIT & LOSS METRICS
# ----------------------------------------------------------------------
def calculate_pl_metrics(processor):
    """
    Calculate Profit & Loss metrics with robust column detection - OPTIMIZED + DEFENSIBLE VERSION.
    """
    try:
        df = processor.df
        processor._income_debug = {}  # for UI debugging

        def _pick_qty_price(species: str, kpmd: bool):
            """
            Helper to find quantity and price columns for each channel.
            species: 'sheep' or 'goat'
            kpmd: True (KPMD buyers) or False (non-KPMD buyers)
            """
            if species == 'sheep' and kpmd:
                qty_exact = ['E1a. How many sheep did you sell to KPMD off-takers  last month?']
                qty_pats = [
                    r'^E1a\..*(how many|number).*(sheep).*sell.*KPMD',
                    r'^E1\..*how many.*sheep.*KPMD'
                ]
                price_exact = ['E1c. What was the average price per sheep last month?']
                price_pats = [
                    r'^E1c\..*(average|avg).*price.*sheep',
                    r'^E1\..*price.*sheep.*KPMD'
                ]
            elif species == 'goat' and kpmd:
                qty_exact = ['E2a. How many goats did you sell to KPMD off-takers  last month?']
                qty_pats = [
                    r'^E2a\..*(how many|number).*(goat).*sell.*KPMD',
                    r'^E2\..*how many.*goat.*KPMD'
                ]
                price_exact = ['E2c. What was the average price per goat last month?']
                price_pats = [
                    r'^E2c\..*(average|avg).*price.*goat',
                    r'^E2\..*price.*goat.*KPMD'
                ]
            elif species == 'sheep' and not kpmd:
                qty_exact = ['E3b. How many sheep did you sell to non-KPMD off-takers  last month?']
                qty_pats = [
                    r'^E3b\..*(how many|number).*(sheep).*sell.*non.*KPMD',
                    r'^E3\..*how many.*sheep.*non'
                ]
                price_exact = ['E3d. What was the average price per sheep last month?']
                price_pats = [
                    r'^E3d\..*(average|avg).*price.*sheep',
                    r'^E3\..*price.*sheep.*non'
                ]
            else:  # goats, non-KPMD
                qty_exact = ['E4b. How many goats did you sell to non-KPMD off-takers  last month?']
                qty_pats = [
                    r'^E4b\..*(how many|number).*(goat).*sell.*non.*KPMD',
                    r'^E4\..*how many.*goat.*non'
                ]
                price_exact = ['E4d. What was the average price per goat last month?']
                price_pats = [
                    r'^E4d\..*(average|avg).*price.*goat',
                    r'^E4\..*price.*goat.*non'
                ]

            qty_col = next((c for c in qty_exact if c in df.columns), None)
            price_col = next((c for c in price_exact if c in df.columns), None)

            if qty_col is None:
                for pat in qty_pats:
                    hits = [c for c in df.columns if re.search(pat, c, flags=re.IGNORECASE)]
                    if hits:
                        qty_col = hits[0]
                        break

            if price_col is None:
                for pat in price_pats:
                    hits = [c for c in df.columns if re.search(pat, c, flags=re.IGNORECASE)]
                    if hits:
                        price_col = hits[0]
                        break

            return qty_col, price_col

        # ---- Revenue sources ----
        sk_qty, sk_price = _pick_qty_price('sheep', True)
        gk_qty, gk_price = _pick_qty_price('goat', True)
        sn_qty, sn_price = _pick_qty_price('sheep', False)
        gn_qty, gn_price = _pick_qty_price('goat', False)

        # Prepare new columns in a batch
        new_columns = {}

        # Sheep KPMD revenue
        if sk_qty and sk_price:
            new_columns['sheep_kpmd_revenue'] = (
                to_num(df[sk_qty]).fillna(0) * to_num(df[sk_price]).fillna(0)
            )
        else:
            new_columns['sheep_kpmd_revenue'] = 0.0

        # Goat KPMD revenue
        if gk_qty and gk_price:
            new_columns['goat_kpmd_revenue'] = (
                to_num(df[gk_qty]).fillna(0) * to_num(df[gk_price]).fillna(0)
            )
        else:
            new_columns['goat_kpmd_revenue'] = 0.0

        # Sheep non-KPMD revenue
        if sn_qty and sn_price:
            new_columns['sheep_non_kpmd_revenue'] = (
                to_num(df[sn_qty]).fillna(0) * to_num(df[sn_price]).fillna(0)
            )
        else:
            new_columns['sheep_non_kpmd_revenue'] = 0.0

        # Goat non-KPMD revenue
        if gn_qty and gn_price:
            new_columns['goat_non_kpmd_revenue'] = (
                to_num(df[gn_qty]).fillna(0) * to_num(df[gn_price]).fillna(0)
            )
        else:
            new_columns['goat_non_kpmd_revenue'] = 0.0

        processor._income_debug['channels'] = [
            {'Channel': 'Sheep KPMD', 'Qty': sk_qty, 'Price': sk_price},
            {'Channel': 'Goat KPMD', 'Qty': gk_qty, 'Price': gk_price},
            {'Channel': 'Sheep Non-KPMD', 'Qty': sn_qty, 'Price': sn_price},
            {'Channel': 'Goat Non-KPMD', 'Qty': gn_qty, 'Price': gn_price},
        ]

        # ------ Feed income ------
        if all(c in df.columns for c in [
            'B6d. At What price did you sell a 15 kg bale last month?',
            'B6e. Number of 15 kg bales sold in the last 1 month?'
        ]):
            new_columns['fodder_revenue'] = (
                to_num(df['B6d. At What price did you sell a 15 kg bale last month?']).fillna(0)
                * to_num(df['B6e. Number of 15 kg bales sold in the last 1 month?']).fillna(0)
            )
        else:
            new_columns.setdefault('fodder_revenue', 0.0)

        # Add all revenue columns at once
        for col_name, col_data in new_columns.items():
            df[col_name] = col_data

        # Calculate total revenue and income components
        revenue_components = [
            'sheep_kpmd_revenue', 'goat_kpmd_revenue',
            'sheep_non_kpmd_revenue', 'goat_non_kpmd_revenue',
            'fodder_revenue'
        ]

        for comp in revenue_components:
            if comp not in df.columns:
                df[comp] = 0.0

        df['total_revenue'] = df[revenue_components].sum(axis=1)
        df['income_kpmd'] = df['sheep_kpmd_revenue'] + df['goat_kpmd_revenue']
        df['income_non_kpmd'] = df['sheep_non_kpmd_revenue'] + df['goat_non_kpmd_revenue']
        df['income_feed'] = df['fodder_revenue']

        # -------- Costs --------
        cost_columns = {}
        cost_component_list = []

        # Feed costs (e.g. purchased feeds)
        if 'Feed_Expenditure' in df.columns:
            cost_columns['feed_costs'] = to_num(df['Feed_Expenditure']).fillna(0)
            cost_component_list.append('feed_costs')

        # Herding costs
        if 'B3b. What was the cost of herding per month (Ksh)?' in df.columns:
            cost_columns['herding_costs'] = to_num(
                df['B3b. What was the cost of herding per month (Ksh)?']
            ).fillna(0)
            cost_component_list.append('herding_costs')

        # --- Veterinary costs (vaccination, treatment, deworming) ---
        vet_costs = []

        # Vaccination: per-animal cost × number vaccinated (if quantity is available)
        vac_cost_col = (
            'D1b. What was the cost of small ruminants vaccination in KSH per animal in the last month?'
        )
        vac_qty_candidates = [
            'D1c. How many small ruminants were vaccinated in the last 1 month?',
            'D1c. Number of small ruminants vaccinated in the last month?'
        ]
        vac_qty_col = next((c for c in vac_qty_candidates if c in df.columns), None)

        if vac_cost_col in df.columns:
            vac_cost_per_animal = to_num(df[vac_cost_col]).fillna(0)
            if vac_qty_col:
                vac_qty = to_num(df[vac_qty_col]).fillna(0)
                cost_columns['vaccination_costs'] = vac_cost_per_animal * vac_qty
            else:
                # Conservative fallback: treat as total cost if qty is missing
                cost_columns['vaccination_costs'] = vac_cost_per_animal
            vet_costs.append('vaccination_costs')

        # Treatment costs (already total in question wording)
        trt_col = 'D3b. What was the total cost of treatment in KSH last month?'
        if trt_col in df.columns:
            cost_columns['treatment_costs'] = to_num(df[trt_col]).fillna(0)
            vet_costs.append('treatment_costs')

        # Deworming costs (already total in question wording)
        dew_col = 'D4a. What was the total of cost of deworming in KSH last month?'
        if dew_col in df.columns:
            cost_columns['deworming_costs'] = to_num(df[dew_col]).fillna(0)
            vet_costs.append('deworming_costs')

        # Aggregate vet costs if any
        if vet_costs:
            for col_name in vet_costs:
                if col_name not in df.columns and col_name in cost_columns:
                    df[col_name] = cost_columns[col_name]
            df['vet_costs'] = df[vet_costs].sum(axis=1)
            cost_component_list.append('vet_costs')

        # --- Transport costs: per-head × quantity sold by channel ---
        transport_costs_series = pd.Series(0.0, index=df.index)

        # Sheep KPMD
        t_sk_col = 'E1h. What was the transport cost to  the market per sheep last month?'
        if t_sk_col in df.columns and sk_qty:
            t_sk = to_num(df[t_sk_col]).fillna(0)
            q_sk = to_num(df[sk_qty]).fillna(0)
            transport_costs_series += t_sk * q_sk

        # Goat KPMD
        t_gk_col = 'E2h. What was the transport cost to  the market per goat last month?'
        if t_gk_col in df.columns and gk_qty:
            t_gk = to_num(df[t_gk_col]).fillna(0)
            q_gk = to_num(df[gk_qty]).fillna(0)
            transport_costs_series += t_gk * q_gk

        # Sheep non-KPMD
        t_sn_col = 'E3i. What was the transport cost to  the market per sheep last month?'
        if t_sn_col in df.columns and sn_qty:
            t_sn = to_num(df[t_sn_col]).fillna(0)
            q_sn = to_num(df[sn_qty]).fillna(0)
            transport_costs_series += t_sn * q_sn

        # Goat non-KPMD
        t_gn_col = 'E4i. What was the transport cost to  the market per goat last month?'
        if t_gn_col in df.columns and gn_qty:
            t_gn = to_num(df[t_gn_col]).fillna(0)
            q_gn = to_num(df[gn_qty]).fillna(0)
            transport_costs_series += t_gn * q_gn

        if (transport_costs_series != 0).any():
            df['transport_costs'] = transport_costs_series
            cost_component_list.append('transport_costs')

        # --- Other costs (fencing, minerals, water, etc.) ---
        other_costs_cols = [
            'B4b. What is the total cost of fencing(Ksh)?',
            'B4b. What is the total monthly cost of use of minerals(Ksh)?',
            'B4b. What is the total monthly cost of catration of small ruminants(Ksh)?',
            'B4b. What is the total monthly cost of hoof trimming(Ksh)?',
            'B4b. What is the total monthly cost of cleaning the pens(Ksh)?',
            'B4b. What is the total monthly cost of ear tagging(Ksh)?',
            'B4b. What is the total monthly cost of water(Ksh)?',
            'B4b. What is the total monthly cost of spraying of acaricides(Ksh)?'
        ]

        existing_other = [c for c in other_costs_cols if c in df.columns]
        if existing_other:
            for c in existing_other:
                df[c] = to_num(df[c]).fillna(0)
            df['other_costs'] = df[existing_other].sum(axis=1)
            cost_component_list.append('other_costs')

        # Add all cost columns we’ve prepared but not yet attached
        for col_name, col_data in cost_columns.items():
            if col_name not in df.columns:
                df[col_name] = col_data

        # Total costs
        if cost_component_list:
            for comp in cost_component_list:
                if comp not in df.columns:
                    df[comp] = 0.0
            df['total_costs'] = df[cost_component_list].sum(axis=1)
        else:
            df['total_costs'] = 0.0

        # Net profit and profit margin
        df['net_profit'] = df['total_revenue'] - df['total_costs']

        valid_revenue = df['total_revenue'] > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            df.loc[valid_revenue, 'profit_margin'] = (
                df.loc[valid_revenue, 'net_profit']
                / df.loc[valid_revenue, 'total_revenue'] * 100
            )
        df['profit_margin'] = df['profit_margin'].fillna(0)

        # Safety net: ensure presence of key columns
        must_have = [
            'sheep_kpmd_revenue', 'goat_kpmd_revenue',
            'sheep_non_kpmd_revenue', 'goat_non_kpmd_revenue',
            'fodder_revenue', 'total_revenue', 'total_costs',
            'net_profit', 'profit_margin', 'income_kpmd',
            'income_non_kpmd', 'income_feed'
        ]

        for c in must_have:
            if c not in df.columns:
                df[c] = 0.0

        processor.df = df

    except Exception as e:
        # Fallback: ensure basic columns exist so UI does not break
        df = processor.df
        default_cols = [
            'total_revenue', 'total_costs', 'net_profit', 'profit_margin',
            'income_kpmd', 'income_non_kpmd', 'income_feed'
        ]
        for col in default_cols:
            if col not in df.columns:
                df[col] = 0.0
        processor.df = df

# ----------------------------------------------------------------------
# FOOD SECURITY & CLIMATE RESILIENCE (STUBS)
# ----------------------------------------------------------------------
def calculate_food_security(processor):
    """
    Prepare food security metrics (currently rCSI only).

    Behaviour:
    - If 'rcsi_30' exists in the dataset, we coerce it to numeric.
    - If it does NOT exist, we leave it missing and let the UI
      report "No rCSI values available" instead of fabricating zeros.
    """
    try:
        df = processor.df.copy()

        if "rcsi_30" in df.columns:
            df["rcsi_30"] = pd.to_numeric(df["rcsi_30"], errors="coerce")

        processor.df = df

    except Exception:
         processor.df = processor.df

def calculate_climate_resilience(processor):
    """Placeholder for climate resilience metrics."""
    try:
        df = processor.df
        if "resilience_score" not in df.columns:
            df["resilience_score"] = 0
        processor.df = df
    except Exception:
        df = processor.df
        if "resilience_score" not in df.columns:
            df["resilience_score"] = 0
        processor.df = df

# ----------------------------------------------------------------------
# WRAPPER
# ----------------------------------------------------------------------
def calculate_all_metrics(processor):
    """
    Calculate all derived metrics for the processor in a single pass.

    Note:
    - Panel metrics are handled inside PanelDataManager, which is instantiated
      in APMTDataProcessor and updates processor.df directly.
    """
    # Herd composition & productivity
    calculate_herd_metrics(processor)

    # Profit & Loss (revenues, costs, margins)
    calculate_pl_metrics(processor)

    # Food security (e.g. rCSI placeholder)
    calculate_food_security(processor)

    # Climate resilience (currently placeholder)
    calculate_climate_resilience(processor)