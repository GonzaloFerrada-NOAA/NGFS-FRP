clear;close;clc

DATE1 = datetime([2025 08 15 00 00 00]);
DATE2 = datetime([2025 09 11 23 00 00]);

PATHIN = '/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/FIRE/NGFS/gridded/netcdf';
PATHRAVE = '/gpfs/f6/drsa-fire3/world-shared/Gonzalo.Ferrada/input/rave/raw';
PATHOUT = '/gpfs/f6/drsa-fire3/scratch/Gonzalo.Ferrada/FIRE/NGFS/gridded/asrave'; mkdir(PATHOUT);

dates = DATE1 : hours(1) : DATE2;

species = ["PM25", "NH3", "SO2","CH4"];
speciesout = ["FRP_MEAN", "FRP_SD", "FRE", "PM25", "NH3", "SO2","CH4"];

for i = 1:numel(dates)
    
    tag = char(dates(i), 'yyyyMMdd_HH');
    f   = [PATHIN '/NGFS_' tag 'Z_0p03.nc']; msg(f)
    
    ngfs = readngfs(f);
    
    tag = char(dates(i), 'yyyyMMddHH');
    f   = findfile([PATHRAVE '/RAVE-HrlyEmiss-3km_v2r0_blend_s' tag '00000_e*.nc']);
    fout= [PATHOUT '/RAVE-HrlyEmiss-3km_v2r0_blend_s' tag '00000.nc'];
    
    % If RAVE file not found for current hour, try to find previous hours (up to 3h before):
    if ~isfile(f)
        f = findfile([PATHRAVE '/RAVE-HrlyEmiss-3km_v2r0_blend_s' char(dates(i) - hours(1), 'yyyyMMddHH') '00000_e*.nc']);
        if ~isfile(f)
            f = findfile([PATHRAVE '/RAVE-HrlyEmiss-3km_v2r0_blend_s' char(dates(i) - hours(2), 'yyyyMMddHH') '00000_e*.nc']);
            if ~isfile(f)
                f = findfile([PATHRAVE '/RAVE-HrlyEmiss-3km_v2r0_blend_s' char(dates(i) - hours(3), 'yyyyMMddHH') '00000_e*.nc']);
            end
        end
    end
    
    rave = readrave(f);
    
    lon_out = double(ncread(f, 'grid_lont'));
    lat_out = double(ncread(f, 'grid_latt'));
    
    % loop through each ngfs point:
    for j = 1:numel(ngfs.lon)
        
        x = findClosestCoord([rave.lon, rave.lat], ngfs.lon(j), ngfs.lat(j));
        
        for s = 1:numel(species)
            factor = rave.(species(s))(x) / rave.FRP_MEAN(x);
            ngfs.(species(s))(j,1) = ngfs.frp_mean(j) * factor;
        end
        
        % Output grid points for later:
        [X(j),Y(j)] = nearestpoint(lon_out,lat_out,ngfs.lon(j),ngfs.lat(j));
        
    end % j
    
    % Latests fixes:
    ngfs.FRE        = ngfs.frp_mean .* 3600;
    ngfs.FRP_MEAN   = ngfs.frp_mean;
    ngfs.FRP_SD     = ngfs.frp_std;
    % ngfs = rmfield(ngfs,{'frp_mean','frp_std'});
    % return
    
    copyfile(f,fout)
    
    % Regrid each ngfs fire emission into same RAVE-like file:    
    for s = 1:numel(speciesout)
        
        data = zeros(size(lon_out));
        
        for j = 1:numel(ngfs.lon)
            data(X(j),Y(j)) = ngfs.(speciesout(s))(j);
        end
        
        ncwrite(fout, speciesout(s), data)
        clear data

    end
    
    clear ngfs rave X Y
    
end



function fullPath = findfile(Pattern)
	d = dir(Pattern);
    if isempty(d)
	    fullPath = 'notexist';
        return
	end
	fullPath = fullfile(d(1).folder, d(1).name);
end
function msg(str)
  disp([datestr(now,'yyyy-mm-dd HH:MM:SS') '    ' str])
end
function rave = readrave(f)
    
    field_names = ["TPM", "FRE", "FRP_MEAN", "PM25", "NH3", "SO2","CH4"];
    
    rave.lon = ncread(f,'grid_lont'); rave.lon = rave.lon - 360;
    rave.lat = ncread(f,'grid_latt'); %rave.lat = rave.lat(1,:);
    
    I = rave.lon >= -138 & rave.lon <= -57 & rave.lat >= 20 & rave.lat <= 54 & ncread(f,'FRP_MEAN') > 0;
    rave.lon = rave.lon(I);
    rave.lat = rave.lat(I);
    
    for i = 1:numel(field_names)
        rave.(field_names(i)) = ncread(f,field_names(i));
        rave.(field_names(i)) = rave.(field_names(i))(I);
    end
    
    rave.lon = rave.lon + 360;
    
end

function ngfs = readngfs(f)
    
    field_names = ["frp_mean", "frp_std"];
    
    lon = ncread(f, 'lon') + 360;
    lat = ncread(f, 'lat');
    [ngfs.lon, ngfs.lat] = ndgrid(lon,lat);
    
    I = ncread(f, 'frp_mean') > 0;
    ngfs.lon = ngfs.lon(I);
    ngfs.lat = ngfs.lat(I);
    
    for i = 1:numel(field_names)
        ngfs.(field_names(i)) = ncread(f,field_names(i));
        ngfs.(field_names(i)) = ngfs.(field_names(i))(I);
    end
    
end

function idx = findClosestCoord(coords, targetLon, targetLat)
    diffs  = coords - [targetLon, targetLat];  % N×2
    dist2  = sum(diffs.^2, 2);                % squared distance
    [~, idx] = min(dist2);                    % index of minimum distance
end

function [x,y] = nearestpoint(lon,lat,qlon,qlat)
    % Calculate the distances between the query point and all data points
    distances   = sqrt((lat(:) - qlat).^2 + (lon(:) - qlon).^2);

    % Find the indexes of the closest points
    [~, idx]    = min(distances);

    % Convert the linear index to row and column indices
    [x, y]      = ind2sub(size(lat), idx);
end