import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.helpers import set_trainable
from utils.losses import *
from models.decoders import *
from models.encoder import Encoder
from utils.losses import CE_loss
from collections import OrderedDict

class Pseudo(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, cons_w_unsup=None, testing=False,
            pretrained=True, use_weak_lables=False, weakly_loss_w=0.4):

        self.num_classes = num_classes
        if not testing:
            assert (sup_loss is not None) and (cons_w_unsup is not None)

        super(Pseudo, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        else:
            self.mode = 'semi'

        # Supervised and unsupervised losses
        if conf['un_loss'] == "KL":
        	self.unsuper_loss = softmax_kl_loss
        elif conf['un_loss'] == "MSE":
            self.unsuper_loss = softmax_mse_loss
        elif conf['un_loss'] == "JS":
        	self.unsuper_loss = softmax_js_loss
        else:
        	raise ValueError(f"Invalid supervised loss {conf['un_loss']}")
        
        self.unsup_loss_w   = cons_w_unsup
        self.sup_loss_w     = conf['supervised_w']
        self.softmax_temp   = conf['softmax_temp']
        self.sup_loss       = sup_loss
        self.sup_type       = conf['sup_loss']
        self.r              = conf['unsup_weight']

        # Use weak labels
        self.use_weak_lables= use_weak_lables
        self.weakly_loss_w  = weakly_loss_w
        # pair wise loss (sup mat)
        self.aux_constraint     = conf['aux_constraint']
        self.aux_constraint_w   = conf['aux_constraint_w']
        # confidence masking (sup mat)
        self.confidence_th      = conf['confidence_th']
        self.confidence_masking = conf['confidence_masking']

        # Create the model
        self.encoder = Encoder(pretrained=pretrained)
        self.teacher_encoder = Encoder(pretrained=pretrained)

        # The main encoder
        upscale             = 8
        num_out_ch          = 2048
        decoder_in_ch       = num_out_ch // 4
        # self.main_decoder   = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)
        self.main_decoder   = VATDecoder(upscale, decoder_in_ch, num_classes=num_classes)

        self.teacher_decoder   = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)

    def freeze_teachers_parameters(self):
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_decoder.parameters():
            p.requires_grad = False

    def forward(self, A_l=None, B_l=None, target_l=None, A_ul=None, B_ul=None, target_ul=None, curr_iter=None, epoch=None):
        # A_l weak aug, A_ul weak aug
        if not self.training:
            # return self.main_decoder(self.encoder(A_l, B_l))
            return self.teacher_decoder(self.teacher_encoder(A_l, B_l))

        # We compute the losses in the forward pass to avoid problems encountered in muti-gpu 

        # Forward pass the labels example
        input_size  = (A_l.size(2), A_l.size(3))
        output_l    = self.main_decoder(self.encoder(A_l, B_l), t_model=self.teacher_decoder)
        # output_l    = self.main_decoder(self.encoder(A_l, B_l))
        if output_l.shape != A_l.shape:
            output_l = F.interpolate(output_l, size=input_size, mode='bilinear', align_corners=True)

        # Supervised loss
        if self.sup_type == 'CE':
            loss_sup = self.sup_loss(output_l, target_l, temperature=self.softmax_temp)
        elif self.sup_type == 'FL':
            loss_sup = self.sup_loss(output_l,target_l) * self.sup_loss_w
        else:
            loss_sup = self.sup_loss(output_l, target_l, curr_iter=curr_iter, epoch=epoch) * self.sup_loss_w

        # If supervised mode only, return
        if self.mode == 'supervised':
            curr_losses = {'loss_sup': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup
            return total_loss, curr_losses, outputs

        # If semi supervised mode
        elif self.mode == 'semi':

            ## freeze teacher model`s parameteres            
            input_size_u    = (A_ul.size(2), A_ul.size(3)) 
            ##### teacher model to produce weak pred for supervised strong pred
            output_ul       = self.main_decoder(self.encoder(A_ul, B_ul), t_model=self.teacher_decoder)
            # output_ul       = self.main_decoder(self.encoder(A_ul, B_ul))
            if output_ul.shape != A_ul.shape:
                output_ul = F.interpolate(output_ul, size=input_size_u, mode='bilinear', align_corners=True)

            # Supervised loss
            if self.sup_type == 'CE':
                loss_unsup = self.sup_loss(output_ul, target_ul, temperature=self.softmax_temp)
            elif self.sup_type == 'FL':
                loss_unsup = self.sup_loss(output_ul,target_ul)
            else:
                loss_unsup = self.sup_loss(output_ul, target_ul, curr_iter=curr_iter, epoch=epoch)

            ###########
            # weight_u    = self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)
            # loss_unsup  = loss_unsup * weight_u    
            outputs = {'sup_pred': output_l, 'unsup_pred': output_ul}
            curr_losses = {'loss_sup': loss_sup}
            curr_losses['loss_unsup'] = loss_unsup
            r = self.r 
            total_loss  = loss_unsup * r  + loss_sup * (1 - r)


            return total_loss, curr_losses, outputs

    def get_backbone_params(self):
        return chain(self.encoder.get_backbone_params(),
                     self.teacher_encoder.get_backbone_params(),
                    )

    def get_other_params(self):
        if self.mode == 'semi':
            return chain(self.encoder.get_module_params(), 
                         self.teacher_encoder.get_module_params(),
                         self.main_decoder.parameters(), 
                         self.teacher_decoder.parameters(),
                        )

        return chain(self.encoder.get_module_params(), self.main_decoder.parameters())
