import argparse
import os

#import hypertune
import trainer.model as model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket',
        help = 'GCS path to output.',
        required = True
    )
    #parser.add_argument(
    #    '--frac',
    #    help = 'Fraction of input to process',
    #    type = float,
    #    required = True
    #)
    #parser.add_argument(
    #    '--maxDepth',
    #    help = 'Depth of trees',
    #    type = int,
    #    default = 5
    #)
    #parser.add_argument(
    #    '--numTrees',
    #    help = 'Number of trees',
    #    type = int,
    #    default = 100
    #)
    parser.add_argument(
        '--projectId',
        help = 'ID (not name) of your project',
        required = True
    )
    parser.add_argument(
        '--job-dir',
        help = 'output directory for model, automatically provided by gcloud',
        required = True
    )
    
    args = parser.parse_args()
    arguments = args.__dict__
    
    #model.PROJECT = arguments['projectId']
    #model.KEYDIR  = 'trainer'
    
    estimator = model.train_and_evaluate()
    
    loc = model.save_model(estimator, 
                           arguments['job_dir'], 'stackoverlow')
    print("Saved model to {}".format(loc))
    
    # this is for hyperparameter tuning
    #hpt = hypertune.HyperTune()
    #hpt.report_hyperparameter_tuning_metric(
    #    hyperparameter_metric_tag='rmse',
    #   metric_value=rmse,
    #    global_step=0)