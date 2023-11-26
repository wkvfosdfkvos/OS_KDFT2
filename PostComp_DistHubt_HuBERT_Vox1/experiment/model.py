
import torch
from transformers import AutoConfig, HubertModel

from exp_lib import module, augment
from exp_lib.util import TorchModuleManager

NUM_TEACHER_HIDDEN_LAYER = 12

class Model(TorchModuleManager):
    def __init__(self, args, num_class):
        super(Model, self).__init__()
        self.test_teacher = True

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

        backend_t = module.ssl_backend.LinearClassifier(
            NUM_TEACHER_HIDDEN_LAYER,
            args['student_hidden_layer_size'], 
            args['embed_size'], 
            weighted_sum=args['weighted_sum'],
        )

        backend_s = module.ssl_backend.LinearClassifier(
            args['student_hidden_layer_num'],
            args['student_hidden_layer_size'], 
            args['embed_size'], 
            weighted_sum=args['weighted_sum'],
        )

        self.add_module('teacher', teacher)
        self.add_module('student', student)
        self.add_module('backend_T', backend_t)
        self.add_module('backend_S', backend_s)

    def tune_kd(self, x):
        '''Perform one iteration FT learning
        '''
        assert self.state == 'train', 'do model.train() first'
        
        self.optimizer.zero_grad()
        
        # teadcher model inference
        with torch.set_grad_enabled(False):
            ssl_label = self.modules['teacher'](x.clone(), output_hidden_states=True).hidden_states
            ssl_label = torch.stack(ssl_label, dim=1)
            clf_label = self.modules['backend_T'](ssl_label)

        # calculate loss
        x = self.modules['student'](x)
        ssl_loss = self.modules['ssl_kd_loss'](x, ssl_label)

        x = self.modules['backend_S'](x)
        clf_loss = self.modules['clf_kd_loss'](x, clf_label)

        loss = ssl_loss + clf_loss

        # back-propagation
        loss.backward()
        self.optimizer.step()

        return ssl_loss.item(), clf_loss.item()
    
    def tune_ft(self, x, label):
        '''Perform one iteration FT learning
        '''
        assert self.state == 'train', 'do model.train() first'
        
        self.optimizer.zero_grad()
        
        if self.is_trainable('teacher'):
            x = self.modules['teacher'](x, output_hidden_states=True)
        else:
            with torch.set_grad_enabled(False):
                x = self.modules['teacher'](x, output_hidden_states=True)

        x = torch.stack(x.hidden_states, dim=1)

        # backend model inference
        x = self.modules['backend_T'](x)

        # calculate loss
        loss = self.modules['ft_loss'](x, label)

        # back-propagation
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def __call__(self, x):
        '''Test inference 
        '''
        return self.teacher_inference(x) if self.test_teacher else self.student_inference(x)
    
    def teacher_inference(self, x):
        assert self.state == 'eval', 'do model.eval() first'
        x = self.modules['teacher'](x, output_hidden_states=True)
        x = torch.stack(x.hidden_states, dim=1)
        x = self.modules['backend_T'](x)
        return x
    
    def student_inference(self, x):
        assert self.state == 'eval', 'do model.eval() first'
        x = self.modules['student'](x)
        x = self.modules['backend_S'](x)
        return x
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_kd_criterion(self, ssl_kd_criterion, clf_kd_criterion):
        self.add_module('ssl_kd_loss', ssl_kd_criterion)
        self.add_module('clf_kd_loss', clf_kd_criterion)

    def set_ft_criterion(self, criterion):
        self.add_module('ft_loss', criterion)
    
    