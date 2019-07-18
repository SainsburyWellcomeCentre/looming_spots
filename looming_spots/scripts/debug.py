def main():
    from looming_spots.db import loom_trial_group
    mtg = loom_trial_group.MouseLoomTrialGroup('CA388_1_marcus')
    trials = mtg.get_trials_of_type('post_test')

if __name__ == '__main__': main()