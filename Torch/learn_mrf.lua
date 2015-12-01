require 'cutorch'
require 'sys'
require 'optim'
require 'nn'
require 'cunn'

require 'util.Tools'
CSLM = require 'CSLM'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training with Lifting')
cmd:text()
cmd:text('Options')
cmd:option('-K', 1, 'Window')
cmd:option('-minc', 1, 'minimum number of appearances for a word')
cmd:option('-input', '../data/mimic_cuis.txt', 'input file')
cmd:text()

-- Parse input params
params = cmd:parse(arg)
params.rundir = cmd:string('experiment', params, {dir=true})

moments, vocab = read_data(params.input, params.K, params.minc)
N = moments:sum(3):sum(2)[1][1][1]
data = moments:div(N)
T = data:size()[2]
K=params.K

print(N)
print(moments[1]:sum())

D = 60
md = CSLM:all_new(K, T, D)

md:optimize()


-----read labels

vocab_train = {}
vocab_train_count = {}

for i, note in ipairs(notes.train_notes_lab) do
    for j, sen in ipairs(note.sentences) do
        for k, lab in ipairs(sen.labels) do
            label = lab.label:match(".*%S")
            if vocab_train_count[label] then
                vocab_train_count[label] = vocab_train_count[label] + 1
            else
                vocab_train_count[label] = 1
                vocab_train[#vocab_train + 1] = label
            end
        end
    end
end

ct = 0
for i, cui in ipairs(vocab) do
    if vocab_train_count[cui] then
        ct = ct + 1
    end
end

----

vocab_dev = {}
vocab_dev_count = {}

for i, note in ipairs(notes.dev_notes_lab) do
    for j, sen in ipairs(note.sentences) do
        for k, lab in ipairs(sen.labels) do
            label = lab.label:match(".*%S")
            if vocab_dev_count[label] then
                vocab_dev_count[label] = vocab_dev_count[label] + 1
            else
                vocab_dev_count[label] = 1
                vocab_dev[#vocab_dev + 1] = label
            end
        end
    end
end

ct = 0
for i, cui in ipairs(vocab) do
    if vocab_dev_count[cui] then
        ct = ct + 1
    end
end
