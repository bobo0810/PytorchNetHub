import visdom 
import numpy as np

class Visualizer():
    def __init__(self, env='main', **kwargs):
        '''
        **kwargs, dict option
        '''
        self.vis = visdom.Visdom(env=env)
        self.index = {}  # x, dict
        self.log_text = ''
        self.env = env
    
    def plot_train_val(self, loss_train=None, loss_val=None):
        '''
        plot val loss and train loss in one figure
        '''
        x = self.index.get('train_val', 0)

        if x == 0:
            loss = loss_train if loss_train else loss_val
            win_y = np.column_stack((loss, loss))
            win_x = np.column_stack((x, x))
            self.win = self.vis.line(Y=win_y, X=win_x, 
                                env=self.env)
                                # opts=dict(
                                #     title='train_test_loss',
                                # ))
            self.index['train_val'] = x + 1
            return 

        if loss_train != None:
            self.vis.line(Y=np.array([loss_train]), X=np.array([x]),
                        win=self.win,
                        name='1',
                        update='append',
                        env=self.env)
            self.index['train_val'] = x + 5
        else:
            self.vis.line(Y=np.array([loss_val]), X=np.array([x]),
                        win=self.win,
                        name='2',
                        update='append',
                        env=self.env)

    def plot_many(self, d):
        '''
        d: dict {name, value}
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def plot(self, name, y, **kwargs):
        '''
        plot('loss', 1.00)
        '''
        x = self.index.get(name, 0) # if none, return 0
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                    win=name,
                    opts=dict(title=name),
                    update=None if x== 0 else 'append',
                    **kwargs)
        self.index[name] = x + 1
    
    def log(self, info, win='log_text'):
        '''
        show text in box not write into txt?
        '''
        pass
