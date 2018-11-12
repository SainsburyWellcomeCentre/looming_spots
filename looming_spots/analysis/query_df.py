def get_all_combos(context_df):
    all_combos = list(set(context_df['context'].values))
    all_combos = [context.strip('r') for context in all_combos]

