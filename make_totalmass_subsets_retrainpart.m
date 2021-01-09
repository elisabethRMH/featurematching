patients=randperm(200);
test_sub={};
eval_sub={};
retrain_sub={};
train_sub={};
for i = 1:20
    test_sub{i}=patients(i*10-9:i*10);
    eval_sub{i}=patients(mod(i*10+11,200):mod(i*10+20-1,200)+1);
    retrain_sub{i}=patients(mod(i*10+1,200):mod(i*10+10-1,200)+1);
    
        
    train_sub{i}=patients(~ismember(patients, test_sub{i}) & ~ismember(patients, eval_sub{i}) & ~ismember(patients, retrain_sub{i}));
    
end
save('totalmass_subsets_retrainpart.mat','*_sub');
