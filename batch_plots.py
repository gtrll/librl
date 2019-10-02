import os
import argparse


def main(exp, style):

    attrs = [
        'MeanSumOfRewards',
        'MinSumOfRewards',
        'MeanRolloutLens',
        'NumSamples',        
        'g_norm',
        'gwocv_norm',
        'decur_norm', 
        'defut_norm',
        'vfn_ev0',
        'vfn_ev1',
        'dyn_ev0',
        'dyn_ev1',
    ]
    
    logdir = os.path.join('/home/robotsrule/Dropbox/exp_results/expic_results', exp)
    os.makedirs(logdir, exist_ok=True)
    for sublogdir in next(os.walk(logdir))[1]:
        ld = os.path.join(logdir, sublogdir)

        for a in attrs:
            print('Plotting {}'.format(a))
            output = os.path.join(ld, '{}.pdf'.format(a))
            flags = ''
            if not style:
                cmd = 'python utils/plot.py --dir {} --curve percentile --value {} {} &'.format(
                    ld, a, flags)
            else:
                cmd = 'python utils/plot.py --dir {} --curve percentile --value {} --style {} {} &'.format(
                    ld, a, style, flags)
            os.system(cmd)
        """Example for setting the limits of axis.

        y_limit = ''
        y_higher = y_lower = None

        # Set y_higher and y_lower...
        if y_higher is not None:
            y_limit += '--y_higher {} '.format(y_higher)
        if y_lower is not None:
            y_limit += '--y_lower {} '.format(y_lower)

        # add y_limit to the plot command.
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, default='.')
    parser.add_argument('--style', type=str, default='')
    args = parser.parse_args()
    main(args.exp, args.style)
