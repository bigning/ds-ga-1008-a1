--------------------------------------------------------
--This script does predition for test_32x32.t7.
--It loads the trained model from results/model.net.
--It also loads the mean and std of training data from 
--results/mean_std, to preprocess the data to make the data
--has 0-mean and 1-norm
-------------------------------------------------------

require 'torch'
require 'image'
require 'nn'
require 'xlua'
require 'optim'

cmd = torch.CmdLine()
cmd:text()
cmd:text("MNIST Testing")
cmd:text()
cmd:text('Options:')
cmd:option('-model_path', 'results/model.net', 'trained model path')
cmd:option('-mean_std', 'results/mean_std', 'training data mean and std')
cmd:option('-data', 'mnist.t7/test_32x32.t7', 'test data')
cmd:option('-type', 'double', 'type:double | float | cuda')
cmd:option('-save_path', 'results/sample_submission.csv', 'save path')
cmd:text()

opt = cmd:parse(arg or {})

print('==> loading model')
net = torch.load(opt.model_path)
mean_std = torch.load(opt.mean_std)
if opt.type == 'cuda' then
    net:cuda()
end

print('==> loading data')
loaded = torch.load(opt.data, 'ascii')
test_data = {data = loaded.data, labels = loaded.labels,
             size = function() return loaded.data:size(1)}

print('==> preprocessing data: normalize globally')
test_data:add(-mean_std.m)
test_data:div(mean_std.s)

print('==> preprocessing data: normalize locally')
neighborhood = image.gaussian1D(7)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

for i = 1, test_data.size() do
    test_data.data[{i, {1}, {}, {}}] = normalization:forward(test_data[{i, {1}, {}, {}}])
end

classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
confusion = optim.ConfusionMatrix(classes)

file = io.open(opt.save_path, 'w')
io.output(file)
io.write('Id,Prediction\n')

print('==> test data forwarding')
net.evaluate()
for t = 1, test_data.size() do
    xlua.progress(t, test_data:size())
    local input = test_data[t]
    if opt.type == 'double' then input = input:double()
    elseif opt.type == 'cuda' then input = input:cuda() end
    local target = test_data.labels[t]

    pred = net:forward(input)
    confusion:add(pred, target)
    io.write(t .. ',' .. pred .. '\n')
end
io.close(file)
print(confusion)

