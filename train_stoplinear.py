from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network_stoplinear import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm
import torch
from torch.autograd import Variable


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step))
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict
    
	
def load_partial(speaker0_step, model_dict):
	pretrained_dict = load_checkpoint(speaker0_step, "transformer")
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict) 
	return model_dict
	       
def main():

    dataset = get_dataset()
    global_step = 0
    #global_step = 200000
    
    m = Model()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.1)
    checkpoint = load_partial(200000, m.state_dict())
    m.load_state_dict(checkpoint)
    #optimizer.load_state_dict(checkpoint['optimizer'])
    
    m = nn.DataParallel(m.cuda())#, device_ids=[7])
    
    m.train()
    
    
    # try to freeze all other parts of the network
    for name, param in m.named_parameters():
        if param.requires_grad:
            if "stop" in name:
                print(name)
                continue
            param.requires_grad = False

    pos_weight = t.FloatTensor([5.]).cuda()
    writer = SummaryWriter()
    
    
    for epoch in range(hp.epochs):
        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
        epoch_loss = 0
        pbar = tqdm(dataloader)
        trainset = len(dataloader)
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)
            
            
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)
                
            character, mel, mel_input, pos_text, pos_mel, _ = data
            label = 0

            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)
            #print("number of ones: ", len(torch.nonzero(stop_tokens))) 
            if len(torch.nonzero(stop_tokens)) < int(stop_tokens.shape[0]*stop_tokens.shape[1]*0.4):
                #print("skipping")
                continue
            
            
            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel, label)

            #mel_loss = nn.L1Loss()(mel_pred, mel)
            #post_mel_loss = nn.L1Loss()(postnet_pred, mel)

            stop_preds = stop_preds.squeeze(-1)
            stop_preds = stop_preds.cuda()
            stop_tokens = stop_tokens.cuda()
            bce_loss = nn.BCEWithLogitsLoss()(stop_preds, stop_tokens)
            loss = bce_loss
            epoch_loss += (loss.item()) * character.shape[0]                 
            loss = Variable(loss, requires_grad=True)

            writer.add_scalars('training_loss',{
                    'bce_loss':bce_loss
                }, global_step)
                
            if global_step % hp.image_step == 1:
                
                for i, prob in enumerate(attn_probs):
                    
                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_%d_0'%global_step, x, i*4+j)
                
                for i, prob in enumerate(attns_enc):
                    num_h = prob.size(0)
                    
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j)
            
                for i, prob in enumerate(attns_dec):

                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j)
                
            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            
            # gradient normalization?
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            
            # Update weights
            optimizer.step()

            if global_step % hp.save_step == 0:
                t.save({'model':m.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_stoplinear_%d.pth.tar' % global_step))
                                
        scheduler.step()
        print("Loss is ", epoch_loss / trainset)
        writer.add_scalars('epoch_loss',{
        'loss':epoch_loss / trainset}, (epoch))

            
            


if __name__ == '__main__':
    main()