import copy
import torch.cuda
import time
import os

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(output, target):
    predict = output.argmax(dim=1, keepdim=True)
    
    corrects = predict.eq(target.view_as(predict)).sum().item()
    
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    
    metric_b = metrics_batch(output=output, target=target)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        
        bs, ncrops, c, h, w = xb.size()
        output = model(xb.view(-1, c, h, w))
        output = output.view(bs, ncrops, -1).mean(1)
        
        # get loss per batch
        loss_b, metric_b = loss_batch(loss_func=loss_func, output=output, target=yb, opt=opt)
        
        # update running loss
        running_loss += loss_b
        
        # update running metric
        if metric_b is not None:
            running_metric += metric_b
        
        # break the loop in case of sanity check
        if sanity_check is True:
            break
    
    # average loss value    
    loss = running_loss / float(len_data)
    
    # average metric value
    metric = running_metric / float(len_data)
    
    return loss, metric

# def train_val(model, params):
#     num_epochs = params['num_epochs']
#     loss_func = params['loss_func']
#     optimizer = params['optimizer']
#     train_dl = params['train_dl']
#     val_dl = params['val_dl']
#     sanity_check = params['sanity_check']
#     scheduler = params['scheduler']
#     path2weights = params['path2weights']
    
#     # history of loss values in each epoch
#     loss_history = {'train' : [],
#                     'val'   : []}
    
#     # history of metric values in each epoch
#     metric_history = {'train' : [],
#                       'val'   : []}
    
#     # initialize
#     best_loss = float('inf')
    
#     for epoch in range(num_epochs):
#         start = time.time()
        
#         # get cuurent learning rate
#         current_lr = get_lr(opt=optimizer)
#         print(f'epoch {epoch}/{num_epochs}, current_lr = {current_lr}')
        
#         # train model on training dataset
#         model.train()
#         train_loss, train_metric = loss_epoch(model=model, loss_func=loss_func, dataset_dl=train_dl,
#                                                 sanity_check=sanity_check, opt=optimizer)
        
#         # collect loss and metric for training dataset
#         loss_history['train'].append(train_loss)
#         metric_history['train'].append(train_metric)
        
#         # evaluate model on validation dataset
#         model.eval()
#         val_loss, val_metric = loss_epoch(model=model, loss_func=loss_func, dataset_dl=val_dl, sanity_check=sanity_check)
        
#         # save best model
        
#         if val_loss < best_loss:
#             best = val_loss
#             best_model_weights = copy.deepcopy(x=model.state_dict())
        
#         # collect loss and metric for validation dataset
#         loss_history['val'].append(val_loss)
#         metric_history['val'].append(val_metric)
        
#         # learning rate schedule
#         scheduler.step()
        
#         end = time.time()
        
#         print(f'train loss : {train_loss:.6f}\t dev loss : {val_loss:.6f}\t accuracy : {100 * val_metric:.2f}\t time : {(end - start):.4f}')
#         print('-' * 10)
        
#         return model, loss_history, metric_history

def train_val(model, params):
    # extract model parameters
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["scheduler"]
    path2weights=params["path2weights"]
    
    # history of loss values in each epoch
    loss_history={
        "train": [],
        "val": [],
    }
    
    # histroy of metric values in each epoch
    metric_history={
        "train": [],
        "val": [],
    }
    
    # 가중치를 저장할 때, 코랩 GPU 오류나서 생략했습니다.
    # a deep copy of weights for the best performing model
    # best_model_wts = copy.deepcopy(model.state_dict())
    
    # initialize best loss to a large value
    best_loss=float('inf')
    
    # main loop
    for epoch in range(num_epochs):
        # check 1 epoch start time
        start_time = time.time()

        # get current learning rate
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        
        # train model on training dataset
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)

        # collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        # evaluate model on validation dataset    
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
        
       
        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # # store weights into a local file
            # torch.save(model.state_dict(), path2weights)
            # print("Copied best model weights!")
        
        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        # learning rate schedule
        lr_scheduler.step()

        print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f, time: %.4f s" %(train_loss,val_loss,100*val_metric, time.time()-start_time))
        print("-"*10) 

    ## load best model weights
    # model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

def createFolder(directory):
    try:
        if not os.path.exists(path=directory):
            os.makedirs(name=directory)
    except OSError:
        print('Error')