dr = '/Users/aidasaglinskas/Desktop/BC-CP-Computational-Profiles/Results/'
mask='/Users/aidasaglinskas/Desktop/BC-CP-Computational-Profiles/Data/brain_mask_bool.nii'
%%
ads=[];
for i=0:13;
fn = sprintf('sub%d-subs-concat-1-testrun2.nii',i);
ds = cosmo_fmri_dataset(fullfile(dr,fn),'mask',mask);
    if isempty(ads);
        ads=ds;
    else
        ads=cosmo_stack({ads,ds});
    end
end
nbrhood=cosmo_spherical_neighborhood(ds,'radius',6);
%% Compute Consistency Scores, Consesus CIDs
e = NaN(size(ads.samples));
cscores=e;ccids=e;
for s = 1:size(ads.samples,1)
    clc;disp(s)
cscore = arrayfun(@(x) 1-length(unique(ads.samples(s,nbrhood.neighbors{x}))) / length(ads.samples(1,nbrhood.neighbors{x})),1:length(ds.samples));

ccid = arrayfun(@(x) mode(ads.samples(s,nbrhood.neighbors{x})),1:length(ds.samples));
ccid(cscore<.80)=0;

cscores(s,:) = cscore;
ccids(s,:) = ccid;
end
disp('done')
%%
%numclust per person
arrayfun(@(s) length(unique(ccids(s,:))),1:14)
csharemat=[];
for s1 = 1:14
for s2 = 1:14
    csharemat(s1,s2) = sum(ismember(unique(ccids(s1,:)),unique(ccids(s2,:))));
end
end
% common clusters mat
add_numbers_to_mat(csharemat)
%% Most Common Cluster Across Subjects
c = arrayfun(@(i) unique(ccids(i,:)),1:14,'UniformOutput',0)
tab = tabulate([c{:}]);[Y I] = sort(tab(:,2),'descend')
tab = tab(I,:)
%% Save Cluster IDs, and Cscores
cds = ds;
odir='/Users/aidasaglinskas/Desktop/BC-CP-Computational-Profiles/Results/cscore_results/'
for s = 1:size(cscores,1)
    
cds.samples=cscores(s,:);
cosmo_map2fmri(cds,fullfile(odir,sprintf('sub%d_cscore.nii',s)));

cds.samples=ccids(s,:);
cosmo_map2fmri(cds,fullfile(odir,sprintf('sub%d_ccid.nii',s)));

cds.samples=ccids(s,:)==4;
cosmo_map2fmri(cds,fullfile(odir,sprintf('sub%d_cluster4.nii',s)));
end
disp('all done')
%%


