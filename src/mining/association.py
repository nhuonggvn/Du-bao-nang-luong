"""Association rule mining module."""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import Tuple


def mine_association_rules(basket_df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mine association rules from transaction data.
    
    Args:
        basket_df: Binary transaction dataframe
        config: Configuration dictionary
        
    Returns:
        Tuple of (frequent_itemsets, rules)
    """
    min_support = config['association']['min_support']
    min_confidence = config['association']['min_confidence']
    min_lift = config['association']['min_lift']
    
    print(f"Mining frequent itemsets with min_support={min_support}...")
    
    # Find frequent itemsets
    frequent_itemsets = apriori(
        basket_df.astype(bool),
        min_support=min_support,
        use_colnames=True
    )
    
    print(f"Found {len(frequent_itemsets)} frequent itemsets")
    
    # Generate association rules
    if len(frequent_itemsets) > 0:
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )
        
        # Filter by lift
        rules = rules[rules['lift'] >= min_lift]
        
        # Sort by lift
        rules = rules.sort_values('lift', ascending=False)
        
        print(f"Generated {len(rules)} association rules")
    else:
        rules = pd.DataFrame()
        print("No rules generated - try lowering min_support")
    
    return frequent_itemsets, rules


def format_rules_for_display(rules: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Format rules for better readability.
    
    Args:
        rules: Rules dataframe from mlxtend
        top_n: Number of top rules to return
        
    Returns:
        Formatted rules dataframe
    """
    if len(rules) == 0:
        return rules
    
    display_rules = rules.head(top_n).copy()
    
    # Convert frozensets to readable strings
    display_rules['antecedents'] = display_rules['antecedents'].apply(
        lambda x: ', '.join(list(x))
    )
    display_rules['consequents'] = display_rules['consequents'].apply(
        lambda x: ', '.join(list(x))
    )
    
    # Select key columns
    display_rules = display_rules[[
        'antecedents', 'consequents', 'support', 
        'confidence', 'lift'
    ]]
    
    # Round metrics
    display_rules['support'] = display_rules['support'].round(4)
    display_rules['confidence'] = display_rules['confidence'].round(4)
    display_rules['lift'] = display_rules['lift'].round(4)
    
    return display_rules


def interpret_rules(rules: pd.DataFrame, top_n: int = 10) -> str:
    """Generate human-readable interpretation of top rules.
    
    Args:
        rules: Association rules dataframe
        top_n: Number of top rules to interpret
        
    Returns:
        Interpretation string
    """
    if len(rules) == 0:
        return "No rules found."
    
    top_rules = rules.head(top_n)
    
    interpretation = f"Top {min(top_n, len(rules))} Association Rules:\n\n"
    
    for idx, rule in top_rules.iterrows():
        antecedents = ', '.join(list(rule['antecedents']))
        consequents = ', '.join(list(rule['consequents']))
        
        interpretation += f"Rule {idx + 1}:\n"
        interpretation += f"  If {antecedents}\n"
        interpretation += f"  Then {consequents}\n"
        interpretation += f"  Support: {rule['support']:.3f} | "
        interpretation += f"Confidence: {rule['confidence']:.3f} | "
        interpretation += f"Lift: {rule['lift']:.3f}\n\n"
    
    return interpretation
