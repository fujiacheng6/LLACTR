import os
import sys
import fuxictr
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
import gc
import argparse
import logging
from pathlib import Path
import nni
import torch
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
    parser.add_argument('--config', type=str, default='../config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='LR_avazu_test', help='The experiment_id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    # llm
    parser.add_argument('--use_nni', type=int, default=0)    
    parser.add_argument('--adding_mode', type=int, default=0)  
    parser.add_argument('--nlp_fields', type=int, default=13)
    # hyper-prarameter
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lambda_llm', type=float, default=0)    
    parser.add_argument('--lambda_loss', type=float, default=0)    
    # unuse
    datasetname='game'
    parser.add_argument('--llm_emb_path2', type=str, default='') # base 
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--distill_loss', type=str, default=None)
    args = vars(parser.parse_args())

    if args['use_nni'] != 0: 
        params_nni = {
            'lambda_llm': 0.1,
            'temperature': 1,
            'weight_decay': 0.0001,
            'lambda_loss': 0.1,
            'llm_emb_path2': "",
        }                  
        optimized_params = nni.get_next_parameter()
        params_nni.update(optimized_params)  
        args['lambda_llm'] = params_nni['lambda_llm']
        args['temperature'] = params_nni['temperature']
        args['weight_decay'] = params_nni['weight_decay']
        args['lambda_loss'] = params_nni['lambda_loss']
        args['llm_emb_path2'] = params_nni['llm_emb_path2']
        print(params_nni)


    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)

    from fuxictr.pytorch import models
    from fuxictr.pytorch.utils import seed_everything
    params['gpu'] = args['gpu']

    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    dataset = params['dataset_id'].split('_')[0].lower()
    try:
        ds = getattr(datasets, dataset)
    except:
        raise RuntimeError('Dataset={} not exist!'.format(dataset))

    feature_encoder = ds.FeatureEncoder(**params)
    
    if params.get("data_format") == 'h5':
        if os.path.exists(feature_encoder.json_file):
            feature_encoder.feature_map.load(feature_encoder.json_file)
        else:
            raise RuntimeError('feature_map not exist!')
    elif params.get('pickle_feature_encoder') and os.path.exists(feature_encoder.pickle_file):
        feature_encoder = feature_encoder.load_pickle(feature_encoder.pickle_file)
    else:
        feature_encoder.fit(**params)


    print('-'*120)
    for key, value in args.items():
        print(f"{key}: {value}")
    print('-'*120)


    
    use_llm_emb = False
    llm_emb = None
    llm_emb2 = None
    train_dot_sum = valid_dot_sum = test_dot_sum = None

    if args['adding_mode'] == 1:
        if args['gpu'] == -1:
            my_device = "cpu"
        else:
            my_device = torch.device("cuda:" + str(args['gpu']))
        print("my_device =", my_device)
        
        if args['llm_emb_path2'] != '':
            s1 = time.time()
            llm_emb2 = torch.load(args['llm_emb_path2'], map_location=my_device).float()
            e1 = time.time()
            print(f"llm_emb: {llm_emb2.shape}, {llm_emb2.device}, cost {e1 - s1} s")

    model_class = getattr(models, params['model'])
    model = model_class(feature_encoder.feature_map, adding_mode=args['adding_mode'], weight_decay=args['weight_decay'], \
        lambda_llm=args['lambda_llm'], temperature=args['temperature'], llm_emb=llm_emb, nlp_fields=args['nlp_fields'],\
        lambda_loss=args['lambda_loss'], llm_emb2=llm_emb2, llm_emb_path2=args['llm_emb_path2'],  **params)
    model.count_parameters() # print number of parameters used in model

    ### original dataloader
    train_gen, valid_gen = datasets.data_generator(feature_encoder, use_llm_emb=use_llm_emb, nlp_fields=args['nlp_fields'],\
        train_dot_sum=train_dot_sum ,valid_dot_sum=valid_dot_sum, stage='train', **params)   
    test_gen = datasets.data_generator(feature_encoder, use_llm_emb=use_llm_emb, nlp_fields=args['nlp_fields'],\
        test_dot_sum=test_dot_sum, stage='test', **params) 

    ## train   
    model.fit_generator(train_gen, validation_data=valid_gen, **params)

    ## valid
    model.load_weights(model.checkpoint)
    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate_generator(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    
    ## test
    logging.info('******** Test evaluation ********')  
    test_result = model.evaluate_generator(test_gen)

    
    # nni
    best_accuracy = {
        "default": test_result['AUC'],
        "logloss": test_result['logloss'],
    }    
    nni.report_final_result(best_accuracy)
    
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    if args['use_nni'] == 0:
        with open(result_filename, 'a+') as fw:
            fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                        ' '.join(sys.argv), experiment_id, params['dataset_id'],
                        "N.A.", print_to_list(valid_result), print_to_list(test_result)))
    else:
        with open(result_filename, 'a+') as fw:
            fw.write(' {},[command] python {},[nni_params] {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                        ' '.join(sys.argv), params_nni, experiment_id, params['dataset_id'],
                        "N.A.", print_to_list(valid_result), print_to_list(test_result))) 
    # time.sleep(3)
    # os.remove(model.checkpoint)
