
import torch
from transformers import AutoConfig, HubertModel

from exp_lib import module, augment
from exp_lib.util import TorchModuleManager

class Model(TorchModuleManager):
    def __init__(self, args, num_class):
        super(Model, self).__init__()
        
        # torch modules
        teacher = HubertModel.from_pretrained(
            args['huggingface_url'],
            from_tf=bool(".ckpt" in args['huggingface_url']),
            config=AutoConfig.from_pretrained(args['huggingface_url']),
            revision="main",
            ignore_mismatched_sizes=False,
        )

        student = module.ssl.StudentHubert_DistHubt(
            args['student_hidden_layer_num'],
            args['student_hidden_layer_size'],
            init_teacher_param=args['init_teacher_idx']
        )

        backend = module.ssl_backend.LinearClassifier(
            args['student_hidden_layer_num'], 
            args['student_hidden_layer_size'], 
            args['embed_size'], 
            weighted_sum=args['weighted_sum'],
            use_TFT=args['use_TFT']
        )

        self.add_module('teacher', teacher)
        self.add_module('student', student)
        self.add_module('backend', backend)

    def tune_kd(self, x):
        '''Perform one iteration FT learning
        '''
        assert self.state == 'train', 'do model.train() first'
        
        self.optimizer.zero_grad()
        
        # teadcher model inference
        with torch.set_grad_enabled(False):
            kd_label = self.modules['teacher'](x.clone(), output_hidden_states=True).hidden_states
            kd_label = torch.stack(kd_label, dim=1)

        # student model inference
        x = self.modules['student'](x)
        
        # calculate loss
        loss = self.modules['kd_loss'](x, kd_label)

        # back-propagation
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def tune_ft(self, x, label):
        '''Perform one iteration FT learning
        '''
        assert self.state == 'train', 'do model.train() first'
        
        self.optimizer.zero_grad()
        
        # student model inference
        if self.is_trainable('student'):
            x = self.modules['student'](x)
        else:
            with torch.set_grad_enabled(False):
                x = self.modules['student'](x)

        # backend model inference
        x = self.modules['backend'](x)

        # calculate loss
        loss = self.modules['ft_loss'](x, label)

        # back-propagation
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def __call__(self, x):
        '''Test inference 
        '''
        assert self.state == 'eval', 'do model.eval() first'
        x = self.modules['student'](x)
        x = self.modules['backend'](x)
        return x
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_kd_criterion(self, criterion):
        self.add_module('kd_loss', criterion)

    def set_ft_criterion(self, criterion):
        self.add_module('ft_loss', criterion)