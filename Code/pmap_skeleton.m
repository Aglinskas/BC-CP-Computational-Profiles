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
nbrhood=cosmo_spherical_neighborhood(ds,'radius',6);it
%% Function goes here 


nvox=length(nbrhood.neighbors)
pmap = rand(14,nvox);
% pmask with actual values
%% Save the files 
cds = ds;
odir='/Users/aidasaglinskas/Desktop/BC-CP-Computational-Profiles/Results/pscore_results/'
numsubs=size(ads.samples,1);
for s = 1:numsubs;

cds.samples=pmap(s,:);
cosmo_map2fmri(cds,fullfile(odir,sprintf('sub%d_pscore.nii',s)));
end
disp('all done')