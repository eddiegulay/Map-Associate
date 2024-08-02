import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from .ingredients import ingredients

te = TransactionEncoder()
te_ary = te.fit_transform(ingredients)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=.02)

def recommend_products(products, rules, top_n=10):
    rules['antecedents'] = rules['antecedents'].apply(lambda x: tuple(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: tuple(x))
    recommendations = rules[rules['antecedents'].apply(lambda x: any(product in x for product in products))]
    recommendations = recommendations.sort_values(by=['confidence', 'lift'], ascending=False)
    top_recommendations = recommendations.head(top_n)
    result = {}
    for _, row in top_recommendations.iterrows():
        for item in row['consequents']:
            result[item] = f"{row['support']*100:.2f}%"
    return result

