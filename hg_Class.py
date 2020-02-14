from hg_blocks import create_hourglass_network, euclidean_loss, bottleneck_block, bottleneck_mobile

class HourglassNet(object):
    def __init__(self, num_classes, num_stacks, num_channels, inres, outres):
        self.num_classes = num_classes
        self.num_stacks = num_stacks
        self.num_channels = num_channels
        self.inres = inres
        self.outres = outres


    def build_model(self, mobile=False, show=True):
        if mobile:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks,
                                                  self.num_channels, self.inres, self.outres, bottleneck_mobile)
        else:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks,
                                                  self.num_channels, self.inres, self.outres, bottleneck_block)
        # show model summary and layer name
        if show:
            self.model.summary()
            model_json = self.model.to_json()

            with open("HourglassNet.json", "w") as json_file:
                json_file.write(model_json)

        return self.model