clear;close;clc

DATE1     = datetime([2025 09 07 00 00 00]);
DATE2     = datetime([2025 09 08 00 00 00]);
PATH_RAVE = '/gpfs/f6/drsa-fire3/world-shared/Gonzalo.Ferrada/opsroot/12km-rave/stmp';
PATH_NGFS = '/gpfs/f6/drsa-fire3/world-shared/Gonzalo.Ferrada/opsroot/12km-ngfs-scaled/stmp';

dates = DATE1 : hours(1) : DATE2;
subset = [-119.5 -118.5 36.5 37.5]; % Garnet fire
ps = [39 39 39 -97.5];

for i = 1:numel(dates)
    
    if hour(dates(i)) ~= 0
        ymd = char(dates(i),'yyyyMMdd');
        tag = char(dates(i),'yyyy-MM-dd_HH.mm.ss');
    else
        ymd = char(dates(i) - day(1),'yyyyMMdd');
        tag = char(dates(i),'yyyy-MM-dd_HH.mm.ss');
    end
    
    msg(tag)
    
    FRAVE = [PATH_RAVE '/' ymd '/rrfs_mpassit_00_v2.1.2/det/mpassit.' tag '.nc']; %msg(FRAVE)
    FNGFS = [PATH_NGFS '/' ymd '/rrfs_mpassit_00_v2.1.2/det/mpassit.' tag '.nc']; %msg(FNGFS)
    
    if 1 == 1
    lon = ncread(FRAVE, 'XLONG');
    lat = ncread(FRAVE, 'XLAT');
    rave.aod = ncread(FRAVE, 'PM2_5');
    ngfs.aod = ncread(FNGFS, 'PM2_5');
    % rave.aod = ncread(FRAVE, 'SMOKE_FINE') .* wrf(FRAVE, 'RHO') ; % kg/kg to ug/m3
    % ngfs.aod = ncread(FNGFS, 'SMOKE_FINE') .* wrf(FNGFS, 'RHO') ; % kg/kg to ug/m3
    % aux = ncread(FRAVE, 'E_BB_OUT_SMOKE_FINE'); aux(aux == 0) = NaN;
    
    ax(1) = subplot(1,2,1); hold on
    spatial(lon, lat, rave.aod(:,:,i), ps, 'GeoTicks', 'off'); colorbar off
    title('RAVE','FontWeight','normal')
    lambgeoticks(ps,10,10)
    
    ax(2) = subplot(1,2,2); hold on
    cb = spatial(lon, lat, ngfs.aod(:,:,i), ps, 'GeoTicks', 'off');
    title('NGFS','FontWeight','normal')
    lambgeoticks(ps,10,10)
    
    % set(ax,'ColorScale','linear','CLim',[0 0.3],'Colormap',hue('cams'))
    set(ax,'ColorScale','log','CLim',[0.1 100],'Colormap',hue('gmao'))
    
    reorganizeaxes(1,2,360,360*(ax(1).YLim/ax(1).XLim),10,1,true)
    reorganizecolorbar(1,2,'bottom',0.8)
    cb.Position(2) = cb.Position(2) - 0.03;
    
    % exportgraphics(gcf,['png/aod550_' tag '.png'],'Resolution',300);close
    exportgraphics(gcf,['png/pm25_sfc_' tag '.png'],'Resolution',300);close
    end
    
    
    
    
    
    
    
    % lon = ncread(FNGFS, 'XLONG');
    % lat = ncread(FNGFS, 'XLAT');
    % % aux = ncread(FNGFS, 'SMOKE_FINE');
    % aux = ncread(FRAVE, 'E_BB_OUT_SMOKE_FINE'); aux(aux == 0) = NaN;
    
    % spatial(lon, lat, aux(:,:,1),'GeoTicks', 'off')
    % xticks(-180:180);yticks(xticks)
    % set(gca,'ColorScale','log')%,'CLim',[0.1 100])
    % axis([-120 -117 36 39])
    % exportgraphics(gcf,'ngfs.png','Resolution',300);close
    % % rave.pm25(i)
    
    if 1 == 0
    [rave.data, rave.z] = getsmokeprofile(FRAVE, subset);
    [ngfs.data, ngfs.z] = getsmokeprofile(FNGFS, subset);
    figure('Position', [1 1 400 400]); hold on
    plot(rave.data, rave.z, '.-', 'LineWidth', 1, 'Color', hue('b'))
    plot(ngfs.data, ngfs.z, '.-', 'LineWidth', 1, 'Color', hue('r'))
    
    set(gca,'XScale', 'linear', 'XLim', [-.3 150], 'YLim', [0 10000], 'TickDir','both')
    pbaspect([0.75 1 1]); xlabel('Smoke emissions (µg m^{-2} s^{-1})'); ylabel('Height m.a.s.l.')
    title([char(dates(i),'yyyy-MM-dd HH') ' Z'], 'FontWeight', 'normal')
    reorganizeaxes(1,1,300,400,1,1)
    exportgraphics(gcf,['png/profile_smoke_conc_' tag '.png'],'Resolution',200);close
    end
    clear rave ngfs
    
    
end





function [out, z] = getsmokeprofile(f, subset)
    
    lon = ncread(f, 'XLONG');
    lat = ncread(f, 'XLAT');
    % aux = ncread(f, 'E_BB_OUT_SMOKE_FINE');
    aux = ncread(f, 'SMOKE_FINE');
    h   = wrf(f, 'Z') .* 10;
    
    I = lon >= subset(1) & lon <= subset(2) & lat >= subset(3) & lat <= subset(4);
    
    for i = 1:size(aux,3)
        
        A      = aux(:,:,i); 
        out(i) = mean(A(I), 'all', 'omitnan' );
        B      = h(:,:,i);
        z(i)   = mean(B(I), 'all', 'omitnan');
    end
    
    out(out == 0) = 2e-3;
    
    
end






