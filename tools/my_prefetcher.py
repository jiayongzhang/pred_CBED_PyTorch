import torch

class data_prefetcher():
    ''' prefetch cbed stacks in same h5 file'''
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_file_loader = next(self.loader)
        except StopIteration:
            self.next_file_loader = None
            return
        #with torch.cuda.stream(self.stream):
        #    self.next_input = self.next_input.cuda(non_blocking=True)
        #    self.next_target = self.next_target.cuda(non_blocking=True)
        #    self.next_input = self.next_input.float()
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        file_loader = self.next_file_loader
        self.preload()
        return file_loader

class h5_prefetcher():
    '''prefetch whole h5 file'''
    def __init__(self, h5_loader):
        self.loader = iter(h5_loader)
        self.stream = torch.cuda.stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
