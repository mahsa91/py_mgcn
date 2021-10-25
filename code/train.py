from __future__ import division
from __future__ import print_function

import time
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score

from utils import load_data, class_accuracy, class_f1, layer_accuracy, layer_f1, dict_to_writer, trains_vals_tests_split
from models import CCN
from tensorboardX import SummaryWriter

# f1 name should be f1_micro or f1_macro
# class_metrics = {'loss': F.nll_loss, 'acc': class_accuracy, 'f1_micro': class_f1, 'f1_macro': class_f1,
#                  'all_f1_micro': f1_score, 'all_f1_macro': f1_score}
class_metrics = {'loss': F.nll_loss, 'all_f1_micro': f1_score, 'all_f1_macro': f1_score}
layer_metrics = {'loss': torch.nn.BCEWithLogitsLoss()}


# Preparing the output of classifier for every layer separately (class numbers should be different over layers)
def prepare_output_labels(outputs, labels, idx):
    tmp_labels = []
    tmp_outputs = []
    max_label = 0
    for i in range(len(labels)):
        tmp_labels.extend(labels[i][idx[i]] + max_label)
        tmp_outputs.extend((outputs[i].max(1)[1].type_as(labels[i]) + max_label)[idx[i]])
        max_label = labels[i].max()

    return tmp_outputs, tmp_labels


# Calculate classification metrics for every layer
def model_stat_in_layer_class(in_classes_pred, in_labels, idx, metrics):
    stats = {}
    for i in range(len(in_classes_pred)):
        if not in_classes_pred[i] is None:
            for metr_name,metr_func in metrics.items():
                if not metr_name in stats:
                    stats[metr_name] = []
                if metr_name[:2] == 'f1':
                    try:
                        stats[metr_name].append(metr_func(in_classes_pred[i][idx[i]], in_labels[i][idx[i]], metr_name.split('f1_')[-1]))
                    except:
                        print("Exception in F1 for class" + str(i) + " for classification")
                        stats[metr_name].append(0)
                elif metr_name[:3] == 'all':
                    continue
                else:
                    stats[metr_name].append(metr_func(in_classes_pred[i][idx[i]], in_labels[i][idx[i]]))
        else:
            for metr_name, metr_func in metrics.items():
                if not metr_name in stats:
                    stats[metr_name] = []
                stats[metr_name].append(0)

    tmp_in_classes_pred, tmp_labels = prepare_output_labels(in_classes_pred, in_labels, idx)
    for metr_name, metr_func in metrics.items():
        if metr_name[:3] == 'all':
            stats[metr_name] = metr_func(tmp_in_classes_pred, tmp_labels, average=metr_name.split('f1_')[-1])

    return stats


# Calculate reconstruction metrics for within and between layer edges (based on the input)
def model_stat_struc(preds, reals, metrics, pos_weights, norms, idx=None):
    stats = {}
    for i in range(len(preds)):
        if reals[i] is None:
            continue
        if 'sparse' in reals[i].type():
            curr_real = reals[i].to_dense()
        else:
            curr_real = reals[i]
        if idx is None:
            final_preds = preds[i]
            final_real = curr_real
        else:
            final_preds = preds[i][idx[i][0],idx[i][1]]
            final_real = curr_real[idx[i][0],idx[i][1]]
        for metr_name,metr_func in metrics.items():
            if not metr_name in stats:
                stats[metr_name] = []
            if metr_name == 'f1':
                try:
                    stats[metr_name].append(metr_func(final_preds, final_real, type = metr_name.split('f1_')[-1]))
                except:
                    print("Exception in F1 for layer" + str(i) + " for structure")
                    stats[metr_name].append(0)
            elif metr_name == 'loss':
                if args.cuda:
                    pw = pos_weights[i] * torch.ones(final_real.size()).cuda()
                else:
                    pw = pos_weights[i] * torch.ones(final_real.size())
                stats[metr_name].append(norms[i]*torch.nn.BCEWithLogitsLoss(pos_weight=pw)(final_preds, final_real))
            else:
                try:
                    stats[metr_name].append(metr_func(final_preds, final_real))
                except:
                    print("Odd problem")

    return stats


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    # Run model
    t1 = time.time()
    classes_pred, bet_layers_pred, in_layers_pred = model(features, adjs)
    # Final statistics
    stats = dict()
    stats['in_class'] = model_stat_in_layer_class(classes_pred, labels, idx_trains, class_metrics)
    # Reconstructing all the within and between layer edges without train and test split (for using part of the edges
    # for reconstruction, idx is needed)
    stats['in_struc'] = model_stat_struc(in_layers_pred, adjs_orig, layer_metrics, adjs_pos_weights, adjs_norms)
    stats['bet_struc'] = model_stat_struc(bet_layers_pred, bet_adjs_orig, layer_metrics, bet_pos_weights, bet_norms)
    # Write to writer
    for name, stat in stats.items():
        dict_to_writer(stat, writer, epoch, name, 'Train')
    # Calculate loss
    loss_train = 0
    for name, stat in stats.items():
        # Weighted loss
        if name == 'in_struc':
            loss_train += wlambda * sum([a * b for a, b in zip(stat['loss'], adj_weight)])
        elif name== 'bet_struc':
            loss_train += wlambda * sum([a * b for a, b in zip(stat['loss'], adj_weight)])
        else:
            loss_train += sum([a * b for a, b in zip(stat['loss'], adj_weight)])

    # Gradients
    loss_train.backward()
    optimizer.step()

    # Validation if needed
    if not args.fastmode:
        model.eval()
        classes_pred, bet_layers_pred, in_layers_pred = model(features, adjs)
        stats = dict()
        stats['in_class'] = model_stat_in_layer_class(classes_pred, labels, idx_vals, class_metrics)

        stats['in_struc'] = model_stat_struc(in_layers_pred, adjs_orig, layer_metrics, adjs_pos_weights, adjs_norms)
        stats['bet_struc'] = model_stat_struc(bet_layers_pred, bet_adjs_orig, layer_metrics, bet_pos_weights, bet_norms)

        for name, stat in stats.items():
            dict_to_writer(stat, writer, epoch, name, 'Validation')
        loss_val = sum([sum(stat['loss']) for stat in stats.values()])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    # Test
    model.eval()
    classes_pred, bet_layers_pred, in_layers_pred = model(features, adjs)
    stats = dict()
    stats['in_class'] = model_stat_in_layer_class(classes_pred, labels, idx_tests, class_metrics)

    for name, stat in stats.items():
        dict_to_writer(stat, writer, epoch, name, 'Test')

    loss_test = sum([sum(stat['loss']) for stat in stats.values()])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    dataset_str = "infra"

    # parameter
    # All combination of parameters will be tested
    # adj_weights contains a list of different configs, for example [[1,1,1]] contains one config with equal weights
    # for a three-layered input
    adj_weights = []
    # parameter, weight of link reconstruction loss function
    wlambdas = [10]
    # parameter, hidden dimension for every layer should be defined
    hidden_structures = [[[32],[32],[32]]]
    # hidden_structures = [[[16], [16], [16]],[[32], [32], [32]],[[64], [64], [64]],[[128], [128], [128]],[[256], [256], [256]]]
    # Learning rate
    lrs = [0.01]
    # test size
    test_sizes = [0.2]

    # Load data
    adjs, adjs_orig, adjs_sizes, adjs_pos_weights, adjs_norms, bet_pos_weights, bet_norms, bet_adjs, bet_adjs_orig, bet_adjs_sizes, \
    features, features_sizes, labels, labels_nclass = load_data(path="../data/" + dataset_str + "/", dataset=dataset_str)
    # Number of layers
    n_inputs = len(adjs)
    # Weights of layers
    adj_weights.append(n_inputs * [1])

    if args.cuda:
        for i in range(n_inputs):
            labels[i] = labels[i].cuda()

            adjs_pos_weights[i] = adjs_pos_weights[i].cuda()
            adjs[i] = adjs[i].cuda()
            adjs_orig[i] = adjs_orig[i].cuda()
            features[i] = features[i].cuda()

        for i in range(len(bet_adjs)):
            if not bet_adjs[i] is None:
                bet_adjs[i] = bet_adjs[i].cuda()
                bet_adjs_orig[i] = bet_adjs_orig[i].cuda()
                bet_pos_weights[i] = bet_pos_weights[i].cuda()
    # Number of runs
    for run in range(10):
        for ts in test_sizes:
            idx_trains, idx_vals, idx_tests = trains_vals_tests_split(n_inputs, [s[0] for s in adjs_sizes], val_size=0.1,
                                                                      test_size=ts, random_state=int(run + ts*100))
            if args.cuda:
                for i in range(n_inputs):
                    idx_trains[i] = idx_trains[i].cuda()
                    idx_vals[i] = idx_vals[i].cuda()
                    idx_tests[i] = idx_tests[i].cuda()
            # Train model
            t_total = time.time()
            for wlambda in wlambdas:
                for adj_weight in adj_weights:
                    for lr in lrs:
                        for hidden_structure in hidden_structures:
                            temp_weight = ['{:.2f}'.format(x) for x in adj_weight]
                            w_str = '-'.join(temp_weight)
                            h_str = '-'.join([','.join(map(str, temp)) for temp in hidden_structure])
                            writer = SummaryWriter('log/' + dataset_str + '_run-'+ str(run)+ '/' + '_lambda-' +
                                                   str(wlambda) + "_adj_w-" + w_str + "_LR-" + str(lr) +
                                                             '_hStruc-' + h_str + '_test_size-' + str(ts))
                            model = CCN(n_inputs=n_inputs,
                                        inputs_nfeat=features_sizes,
                                        inputs_nhid=hidden_structure,
                                        inputs_nclass=labels_nclass,
                                        dropout=args.dropout)
                            optimizer = optim.Adam(model.parameters(),
                                                   lr=lr, weight_decay=args.weight_decay)
                            if args.cuda:
                                model.cuda()
                            for epoch in range(args.epochs):
                                train(epoch)
                                # Testing
                                test()
                            print("Optimization Finished!")
                            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


