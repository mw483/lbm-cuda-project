clear
close all

%設定・・・要変更
cd_position = ""; % 粒子ファイルのフォルダ
x_grid    = 320; %grid size
y_grid    = 320; %grid size
z_grid    = 200; %grid size
resolut   = 2;   %[m]
pout      = 125; %粒子の生成間隔(step)
dt        = 0.008; %timestep 
start_num = 60; %最初のファイル番号
data_num  = 600; %最後のファイル番号
dout      = 1; %画像出力の間隔
% frga
id_read         = 0; %id読み込み･･･0=off/1=on
position_read   = 1; %position読み込み･･･0=off/1=on
id_on           = 0; %idピックアップ･･･0=off/1=on
mapfrag         = 0; %Map出力･･･0=off/1=on
png_out         = 1; %画像出力…0=off/1=on
pngsave_on      = 1; %画像保存…0=off/1=on
gifsave_on      = 1; %Gif保存…0=off/1=on
Out_PNG         = "G:\ParticleValidation\case9\Figure\PNG"; %画像保存ディレクトリ
Out_GIF         = "G:\ParticleValidation\case9\Figure\GIF"; %Gif保存ディレクトリ
dim_flag        = 3; %filename 2=2D/3=3D_position-
filename        = 'debug_SGS.gif'; %gif filename
filetype        = 1; % 1=.bin/2=.csv
id_pick         = 500;


if mapfrag == 1
    %建物 3D or 2D
    X = 1:resolut:x_grid*resolut;
    Y = 1:resolut:y_grid*resolut;
    Z_def = readmatrix(fname_map);  
    Z = Z_def(1:x_grid,1:y_grid);
%     Z(Z==0) = NaN;
    if dim_flag == 3
    %3D
    surf(X,Y,flipud(Z),'LineStyle','none')
%     colormap([255/255 235/255 205/255]) 
    colormap([0/255 0/255 255/255]) 
    view(35,50) %視点変更
    
    camlight('left','infinite')
    ax = gca;
        ax.Color = [235/255 235/255 255/255];
 
    elseif dim_flag == 2
    %2D
    pcolor(X,Y,flipud(Z));
    shading flat
    else
    exit
    end 
    hold on
end
n = 4;
%以下変更必要なし
%粒子の可視化
for i = start_num:dout:data_num-1
if id_read ==1
    fname_particle = strcat(cd_id,"\index0-",num2str(i),".bin");
    fileID = fopen(fname_particle);
    A = fread(fileID,Inf,'int');
    fclose(fileID);
    n = numel(A);
    id  = A(2:n);
end    
if position_read ==1
    fname_particle = strcat(cd_position,"\position0-",num2str(i),".bin");
    fileID = fopen(fname_particle);
    A = fread(fileID,Inf,'float');
    fclose(fileID);
    n = numel(A);
    position  = [A(2:3:n-2) A(3:3:n-1) A(4:3:n)];
end
if id_read == 1 && position_read == 1
    particle = [id position];
end
if png_out == 1
    if id_on == 1
        particle_ID = [];
        for it = 1:(n-1)/3
            if particle(it,1) == id_pick
                particle_ID     = vertcat(particle_ID,particle(it,:));
            end
        end
        sc =scatter3(particle_ID(:,2),y_grid*resolut-particle_ID(:,3),particle_ID(:,4),100);
            sc.Marker     = '.';
            sc.MarkerEdgeColor = [205/255 205/255 205/255];
            sc.MarkerFaceColor = [255/255 255/255 255/255];
            sc.MarkerEdgeAlpha = 0.5;
            sc.MarkerFaceAlpha = 0.5;
            sc.LineWidth       = 100;
            axis([0 x_grid*resolut 0 y_grid*resolut 0 z_grid*resolut])
            daspect([1 1 1])
        f = gcf; 
            f.Units = 'centimeters';
            f.Position = [0 0 20 20];
        set(gca, 'LooseInset', get(gca, 'TightInset'));
        tt = i * pout * dt;
        h = fix(tt/3600);
        m = fix((tt-3600*h)/60);
        s = tt-3600*h - 60 * m;
        nm = sprintf('%02d',m);
        ns = sprintf('%02d',s);
        t = title(strcat(num2str(h),':',nm,':',ns));
            t.FontSize = 16;
        drawnow
    else
        sc =scatter3(position(:,1),y_grid*resolut-position(:,2),position(:,3),10);
            sc.Marker     = 'o';
%             sc.MarkerEdgeColor = [205/255 205/255 205/255];
%             sc.MarkerFaceColor = [255/255 255/255 255/255];
%             sc.MarkerEdgeColor = [105/255 105/255 105/255];
%             sc.MarkerFaceColor = [255/255 255/255 255/255];
            sc.MarkerEdgeColor = 'none';
            sc.MarkerFaceColor = 'flat';
            sc.MarkerEdgeAlpha = 0.5;
            sc.MarkerFaceAlpha = 0.5;
%             sc.LineWidth       = 0.001;
            sc.LineWidth       = 1;
            axis([0 x_grid*resolut 0 y_grid*resolut 0 z_grid*resolut])
            daspect([1 1 1])
        f = gcf; 
            f.Units = 'centimeters';
            f.Position = [0 0 20 20];
        set(gca, 'LooseInset', get(gca, 'TightInset'));
        ax = gca;
            ax.Color = 'w';
        tt = i * pout * dt;
        h = fix(tt/3600);
        m = fix((tt-3600*h)/60);
        s = tt-3600*h - 60 * m;
        nm = sprintf('%02d',m);
        ns = sprintf('%02d',s);
        t = title(strcat(num2str(h),':',nm,':',ns));
            t.FontSize = 16;
%         
%         view(0,0)
        xlabel("X [m]");
        ylabel("Y [m]");
        zlabel("Z [m]")
        drawnow
    end
end
% 保存
if pngsave_on == 1
    if dim_flag == 3
        Dim = "3D";
    elseif dim_flag == 2
        Dim = "2D";
    else
        Dim = "";
    end
    hgexport(gcf,strcat(Out_PNG,"\",Dim,"position-",num2str(i)),hgexport('factorystyle'),'Format','png')
end
if gifsave_on == 1
    %make gif file_b
    frame = getframe(f);
    i = i + 1;
    im{i} = frame2im(frame);
    [A,map] = rgb2ind(im{i},256);
    
    if i == start_num + 1
    imwrite(A,map,strcat(Out_GIF,"\",filename),'gif','LoopCount',Inf,'DelayTime',0);
    else
    imwrite(A,map,strcat(Out_GIF,"\",filename),'gif','WriteMode','append','DelayTime',0);
    end
end
%particleのみ消去
if png_out == 1
    if i < data_num
        delete(sc)
    end
end
end