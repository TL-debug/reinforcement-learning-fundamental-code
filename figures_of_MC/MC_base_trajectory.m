% 强化学习 无模型的 policy iteration

%% ---------------- 加强学习例题-走迷宫 -------------------- %%
% - 强化学习走迷宫的流程 - 
% 1 policy evaluation:
%   计算所有状态下对应所有action的qvalue，这种action的行动数由自己定义。
% 2 policy improment
%   若qvlue的值相同，可以使用其他原则来选择action，（能量最小？路径最短？还是探索长度最短？）
%
% ---- 本人的理解 ----
% 1. q_pi (s,a) 既对应每个episode的reward的和
%    a) 第一个state采取所有的action，后续的state根据policy确定，由此记录对应的reward
%    b) q_pi的大小为 number of state times number of action
%    ps: bian'li


clear;clc;
% ---- 规则
r_boundary = -1 ; r_forbiden = -10 ; r_traget = 1 ; r_pass = 0; gamma = 0.9;
% 走到边界后返回当前网格

% ---- 生成迷宫（state的编号）
%  3 | 6 | 9
%  2 | 5 | 8
%  1 | 4 | 7
matrix_maze = [1 3 2;0 3 0;0 0 0];% 0:通行区域(白) 1:起点(黄) 2:终点(绿) 3:禁止区域(红) 
color_map   = [1 1 1;1 1 0;0 1 0;1 0 0];% 各区域的颜色
arrow_mod   = [0.45,0.65;0.57,0.5;0.45,0.3;0.25,0.5;0.45,0.5];% 绘制箭头的修正量

matrix_pi0  = randi(5, 3, 3); % 初始策略
action2 = {'↑','→','↓','←','o'};%policy a1=1 a2=2 a3=3 a4=4 a5=5
ng = size(matrix_pi0,1);% 迷宫中向左或向右跳的网格数
% - policy and action value matrixs (通过读取pi_act 来绘图)
pi_act = zeros(size(matrix_pi0,1)*size(matrix_pi0,2),length(action2));
reword = pi_act;

% policy of deterministic action（行对应的网格标号，列表示在对应action序号上行动）
tmp_pi = reshape(matrix_pi0,[],1);
for j1 = 1:length(tmp_pi)
    pi_act(j1, tmp_pi(j1)) = 1;
end

size_maze = size(matrix_maze);


% ----- 绘制迷宫和加速度 -----（子程序）
for j2 = 1:length(tmp_pi)

    indx = floor(j2/(sqrt(length(tmp_pi))+0.1))+1;
    indy = j2 - sqrt(length(tmp_pi))*(indx -1);

    % 网格
    patch([indx indx+1 indx+1 indx],[indy indy indy+1 indy+1],color_map(matrix_maze(indy,indx)+1,:),'EdgeColor','k','LineWidth',2);hold on;
    % 加速度(pi)
    for j3 = 1:size(pi_act,2)
        if ( pi_act(j2,j3) ~=0 )
            text(indx+arrow_mod(j3,1),indy+arrow_mod(j3,2),action2(j3),'FontSize',30);
        end
    end
end
axis equal;axis off;


% - 路径追踪 - 
%1）更具当前策略和输入的步长，便利所有state返回指定补偿的路径
%2）根据输入的 steps 返回路径并计算q(s,a)
steps = 20;
matrix_SA = zeros(2*length(pi_act),steps);% 遍历所有状态列是episode length（奇数行为state 偶数行reword）;行/2 state数
for js = 1:length(pi_act)
    matrix_SA(1+ 2*(js-1),1) = js;% 初始化位置
end
% 网格的顶点和边界编号
vert_ll = 1;vert_ul = ng;vert_ur = ng^2;vert_lr = ng^2-ng+1;
edge_l = 2:ng-1;edge_u = ng*(2:ng-1);
edge_r = (2:ng-1)+(ng-1)*ng;edge_d = 1+ng*(1:ng-2);
forb_s = find(matrix_maze == 3);
targ_s = find(matrix_maze == 2);
% 步进向量
vert_po = [1 ng -1 -ng 0];%  a1 a2 a3 a4 a5

for js = 1:length(pi_act)
    
    tmp_sa = matrix_SA(2*js-1:2*js,:);

    for ja = 1:steps-1
        cur_state = tmp_sa(1,ja);
        % 是否为顶点(顺时针：左下 左上 右上 右下)
        if (  cur_state == vert_ll ) 
            tmp_pol = pi_act(cur_state,:)';
            
            if ( tmp_pol(3) == 1 || tmp_pol(4) == 1) % 撞边界返回
                tmp_sa(1,ja+1) = cur_state;% 状态
                tmp_sa(2,ja+1) = r_boundary;% 奖励
            else % 向其他方向运动或保持
                tmp_pol(3) = 0;tmp_pol(4) = 0;
                tmp_sa(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
                % 判断移动后的位置
                if (max(ismember(tmp_sa(1,ja+1),forb_s)) == 1 )% 禁区
                    tmp_sa(2,ja+1) =  r_forbiden;
                elseif (max(ismember(tmp_sa(1,ja+1),targ_s)) == 1) % 目的地
                    tmp_sa(2,ja+1) =  r_traget;
                else % 通行
                    tmp_sa(2,ja+1) = r_pass;
                end
            end
            
        elseif ( cur_state == vert_ul ) % 左上顶点
            tmp_pol = pi_act(cur_state,:)';

            if ( tmp_pol(1) == 1 || tmp_pol(4) == 1) % 撞边界返回
                tmp_sa(1,ja+1) = cur_state;% 状态
                tmp_sa(2,ja+1) = r_boundary;% 奖励
            else % 向其他方向运动或保持
                tmp_pol(1) = 0;tmp_pol(4) = 0;% 更新策略
                tmp_sa(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
                % 判断移动后的位置
                if (max(ismember(tmp_sa(1,ja+1),forb_s)) == 1 )% 禁区
                    tmp_sa(2,ja+1) =  r_forbiden;
                elseif (max(ismember(tmp_sa(1,ja+1),targ_s)) == 1) % 目的地
                    tmp_sa(2,ja+1) =  r_traget;
                else % 通行
                    tmp_sa(2,ja+1) = r_pass;
                end
            end

        elseif ( cur_state == vert_ur )
            tmp_pol = pi_act(cur_state,:)';% 读取当前状态下的策略

            if ( tmp_pol(1) == 1 || tmp_pol(2) == 1) % 撞边界返回
                tmp_sa(1,ja+1) = cur_state;% 状态
                tmp_sa(2,ja+1) = r_boundary;% 奖励
            else % 向其他方向运动或保持
                tmp_pol(1) = 0;tmp_pol(2) = 0;% 更新策略
                tmp_sa(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
                % 判断移动后的位置
                if (max(ismember(tmp_sa(1,ja+1),forb_s)) == 1 )% 禁区
                    tmp_sa(2,ja+1) = r_forbiden;
                elseif (max(ismember(tmp_sa(1,ja+1),targ_s)) == 1) % 目的地
                    tmp_sa(2,ja+1) = r_traget;
                else % 通行
                    tmp_sa(2,ja+1) = r_pass;
                end
            end

        elseif ( cur_state == vert_lr )
            tmp_pol = pi_act(cur_state,:)';% 读取当前状态下的策略

            if ( tmp_pol(2) == 1 || tmp_pol(3) == 1) % 撞边界返回
                tmp_sa(1,ja+1) = cur_state;% 状态
                tmp_sa(2,ja+1) = r_boundary;% 奖励
            else % 向其他方向运动或保持
                tmp_pol(2) = 0;tmp_pol(3) = 0;% 更新策略
                tmp_sa(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
                % 判断移动后的位置
                if (max(ismember(tmp_sa(1,ja+1),forb_s)) == 1 )% 禁区
                    tmp_sa(2,ja+1) = r_forbiden;
                elseif (max(ismember(tmp_sa(1,ja+1),targ_s)) == 1) % 目的地
                    tmp_sa(2,ja+1) = r_traget;
                else % 通行
                    tmp_sa(2,ja+1) = r_pass;
                end
            end

        % 是否为边界(左边界 上边界 右边界 下边界)
        elseif ( ismember(cur_state, edge_l) )
            tmp_pol = pi_act(cur_state,:)';% 读取当前状态下的策略

            if ( tmp_pol(4) == 1 ) % 撞边界返回
                tmp_sa(1,ja+1) = cur_state;% 状态
                tmp_sa(2,ja+1) = r_boundary;% 奖励
            
            else % 向其他方向运动或保持
                tmp_pol(4) = 0;
                tmp_sa(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
                % 判断移动后的位置
                if (max(ismember(tmp_sa(1,ja+1),forb_s)) == 1 )% 禁区
                    tmp_sa(2,ja+1) = r_forbiden;
                elseif (max(ismember(tmp_sa(1,ja+1),targ_s)) == 1) % 目的地
                    tmp_sa(2,ja+1) = r_traget;
                else % 通行
                    tmp_sa(2,ja+1) = r_pass;
                end
            end

        elseif ( ismember(cur_state, edge_u) )
            tmp_pol = pi_act(cur_state,:)';% 读取当前状态下的策略

            if ( tmp_pol(1) == 1 ) % 撞边界返回
                tmp_sa(1,ja+1) = cur_state;% 状态
                tmp_sa(2,ja+1) = r_boundary;% 奖励
            
            else % 向其他方向运动或保持
                tmp_pol(1) = 0;
                tmp_sa(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
                % 判断移动后的位置
                if (max(ismember(tmp_sa(1,ja+1),forb_s)) == 1 )% 禁区
                    tmp_sa(2,ja+1) = r_forbiden;
                elseif (max(ismember(tmp_sa(1,ja+1),targ_s)) == 1) % 目的地
                    tmp_sa(2,ja+1) = r_traget;
                else % 通行
                    tmp_sa(2,ja+1) = r_pass;
                end
            end

        elseif ( ismember(cur_state, edge_r) )
            tmp_pol = pi_act(cur_state,:)';% 读取当前状态下的策略

            if ( tmp_pol(2) == 1 ) % 撞边界返回
                tmp_sa(1,ja+1) = cur_state;% 状态
                tmp_sa(2,ja+1) = r_boundary;% 奖励
            
            else % 向其他方向运动或保持
                tmp_pol(2) = 0;
                tmp_sa(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
                % 判断移动后的位置
                if (max(ismember(tmp_sa(1,ja+1),forb_s)) == 1 )% 禁区
                    tmp_sa(2,ja+1) = r_forbiden;
                elseif (max(ismember(tmp_sa(1,ja+1),targ_s)) == 1) % 目的地
                    tmp_sa(2,ja+1) = r_traget;
                else % 通行
                    tmp_sa(2,ja+1) = r_pass;
                end
            end


        elseif ( ismember(cur_state, edge_d) )
            tmp_pol = pi_act(cur_state,:)';% 读取当前状态下的策略

            if ( tmp_pol(3) == 1 ) % 撞边界返回
                tmp_sa(1,ja+1) = cur_state;% 状态
                tmp_sa(2,ja+1) = r_boundary;% 奖励
            
            else % 向其他方向运动或保持
                tmp_pol(3) = 0;
                tmp_sa(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
                % 判断移动后的位置
                if (max(ismember(tmp_sa(1,ja+1),forb_s)) == 1 )% 禁区
                    tmp_sa(2,ja+1) = r_forbiden;
                elseif (max(ismember(tmp_sa(1,ja+1),targ_s)) == 1) % 目的地
                    tmp_sa(2,ja+1) = r_traget;
                else % 通行
                    tmp_sa(2,ja+1) = r_pass;
                end
            end

        % 中间区域的点
        else
            tmp_pol = pi_act(cur_state,:)';% 读取当前状态下的策略
            tmp_sa(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
            % 判断移动后的位置
            if (max(ismember(tmp_sa(1,ja+1),forb_s)) == 1 )% 禁区
                tmp_sa(2,ja+1) = r_forbiden;
            elseif (max(ismember(tmp_sa(1,ja+1),targ_s)) == 1) % 目的地
                tmp_sa(2,ja+1) = r_traget;
            else % 通行
                tmp_sa(2,ja+1) = r_pass;
            end

        end
    end

    % - 计算完成的路径保存
    matrix_SA(2*js-1:2*js,:) = tmp_sa;

end


% 计算每个state的reward
% matrix_SA 每个episode的中的 state 和 reward 
% 根据q_value 更新 policy



% ----- 对于每个state对每个action都计算n的长度 
ng = size(matrix_pi0,1);
% 网格的顶点和边界编号
vert_ll = 1;vert_ul = ng;vert_ur = ng^2;vert_lr = ng^2-ng+1;     % 顶点
edge_l = 2:ng-1;edge_u = ng*(2:ng-1);                            % 边界
edge_r = (2:ng-1)+(ng-1)*ng;edge_d = 1+ng*(1:ng-2);              % 边界
forb_s = find(matrix_maze == 3);targ_s = find(matrix_maze == 2); % 禁区
VERT_EDGE = {vert_ll,vert_ul,vert_ur,vert_lr,...
             edge_l,edge_u,edge_r,edge_d,forb_s,targ_s};% 顶点；边界；禁区；目标（矩阵中个元素的顺序）

% 规则（奖励或惩罚的分数）
r_boundary = -1 ; r_forbiden = -10 ; r_traget = 1 ; r_pass = 0; gamma = 0.9;
REW = {r_boundary,r_forbiden,r_traget,r_pass,gamma};

% - 最大迭代数
itm = 6;
iti = 1;
% 
steps = 20;
% - 初始化保存 (---记录卡：绘制机器人路径----)
mat_sta = zeros(length(tmp_pi)*length(action2),steps);% 每个action记录长度为n的trajectory
mat_act = mat_sta;% 记录对应的action
mat_rew = mat_sta;% 记录每一步的奖励



while( iti <= itm )


    % - 针对每个state都生成length(action)*steps的trajectory
    for j1 = 1:length(tmp_pi)

        tmp_sta = zeros(length(action2),steps);
        tmp_act = tmp_sta;
        tmp_rew = tmp_sta;

        % 初始化
        tmp_sta(:,1) = j1*ones(length(action2),1);
        tmp_act(:,1) = (1:5)';

        % 根据策略生成trajectory
        for j2 = 1:length(action2)
            % 输入参数更新
            tmp_sar = [tmp_sta(j2,:);tmp_act(j2,:);tmp_rew(j2,:)];
            % 更改策略（按照遍历action的policy进行路径生成）
            tmp_pi_act = pi_act;tmp_pi_act(j1,:) = 0;tmp_pi_act(j1,j2) = 1;

            [sta_act_rew] = TRAJ(tmp_sar,tmp_pi_act,VERT_EDGE,REW,steps,size_maze);% 返回状态，运动和奖励


            % 将返回数据保存到次要矩阵中
            tmp_sta(j2,:) = sta_act_rew(1,:);
            tmp_act(j2,:) = sta_act_rew(2,:);
            tmp_red(j2,:) = sta_act_rew(3,:);

        end

        % 将返回的数据保存到主要矩阵中
        mat_sta(1+length(action2)*(j1-1):5+length(action2)*(j1-1),:) = tmp_sta;
        mat_act(1+length(action2)*(j1-1):5+length(action2)*(j1-1),:) = tmp_act;
        mat_red(1+length(action2)*(j1-1):5+length(action2)*(j1-1),:) = tmp_red;

    end


    % ---- policy improment ---- %
    % - q value -
    % 1.episode当中涉及其他的state的reward时，有其他state出发的机器人经过这里，
    %   那么识别包含state-reward情况，再统计所有state和reward。（如何快速的识别？）
    % 2.对一个episode仅计算一个state的qvalue
    % 生成向量（矩阵）qvalue前的系数向量
    coe_qvalue = [0,logspace(0,log10(gamma^18),19)]';
    qvalue = zeros(size(mat_red(:,1)));
    for j5 = 1:size(mat_red,1)
        qvalue(j5) = mat_red(j5,:)*coe_qvalue;
    end
    % - update policy (pi_act) -
    pi_act = 0*pi_act;
    for j5 = 1:size(pi_act,1)
        % - 位置指标 -
        tmp_ind = [1:5]+5*(j5-1);
        % - 更新policy -
        max_ind = find(qvalue(tmp_ind) == max(qvalue(tmp_ind)));
        % - deterministic process （根据相同q值的数量随机选择一个action,pi_act重置为0）
        random_num = max_ind(randperm(numel(max_ind), 1));%针对这个值更新pi_act
        pi_act(j5,max_ind(randperm(numel(max_ind), 1))) = 1;
        % - stochastic process （根据5个action的q值的更新策略）
    end

    % - 保存每次迭代的qvlaue和policy进行后续绘图 - %

    % ----- 绘制迷宫和加速度 -----（子程序）
    for j2 = 1:length(tmp_pi)
        indx = floor(j2/(sqrt(length(tmp_pi))+0.1))+1;
        indy = j2 - sqrt(length(tmp_pi))*(indx -1);

        % 网格
        patch([indx indx+1 indx+1 indx],[indy indy indy+1 indy+1],color_map(matrix_maze(indy,indx)+1,:),'EdgeColor','k','LineWidth',2);hold on;
        % 加速度(pi)
        for j3 = 1:size(pi_act,2)
            if ( pi_act(j2,j3) ~=0 )
                text(indx+arrow_mod(j3,1),indy+arrow_mod(j3,2),action2(j3),'FontSize',30);
            end
        end
    end
    % - 展示迭代次数
    title(['iteration = ',num2str(iti)],'fontweight','bold','fontsize',14);

    axis equal;axis off;drawnow;pause(0.5);

    % - 绘制迭代过程中的策略图 - %


    % - 停止迭代阈值/执行有限次迭代 -
    iti = iti + 1;

end






%% ---- 子程序
% - 根据当前的policy生成长度为n的trajectory
% 1)针对每个action生成长度为steps的trajectory，计算q(s,a)
% 2)每个state的第一个action都要选择，剩下的action根据policy来更新state
% 3)action是一个deterministic过程
function [sta_act_rew] = TRAJ(tmp_sar,pi_act,VERT_EDGE,REW,steps,size_maze)% policy 边界顶点信息 奖励惩罚规则 episode的长度 迷宫大小

% 迷宫的顶点与边界，禁止区域，目标区域
vert_ll = cell2mat(VERT_EDGE(1));vert_ul = cell2mat(VERT_EDGE(2));
vert_ur = cell2mat(VERT_EDGE(3));vert_lr = cell2mat(VERT_EDGE(4));
edge_l  = cell2mat(VERT_EDGE(5));edge_u  = cell2mat(VERT_EDGE(6));
edge_r  = cell2mat(VERT_EDGE(7));edge_d  = cell2mat(VERT_EDGE(8));
forb_s  = cell2mat(VERT_EDGE(9));targ_s  = cell2mat(VERT_EDGE(10));
r_boundary = cell2mat(REW(1)) ;r_forbiden = cell2mat(REW(2)) ;
r_traget = cell2mat(REW(3)) ; r_pass = cell2mat(REW(4)); 
% gamma = cell2mat(REW(5));

% 状态步进向量（ng表示行数）
ng = size_maze(1);
vert_po = [1 ng -1 -ng 0];% a1 a2 a3 a4 a5
markact = (1:5)';

for ja = 1:steps - 1% 生成trajectory (当j3=1时act为遍历输入值，j3=2:step-1时act为state对应值)

    % 当前状态
    cur_state = tmp_sar(1,ja);  % state action reward = sar
    tmp_pol = pi_act(cur_state,:)';
    % 是否为顶点(顺时针：左下 左上 右上 右下)
    if (  cur_state == vert_ll )
        if ( tmp_pol(3) == 1 || tmp_pol(4) == 1) % 撞边界返回
            tmp_sar(1,ja+1) = cur_state;% 状态
            tmp_sar(2,ja+1) = pi_act(cur_state,:)*markact;% 行为(deterministic)
            tmp_sar(3,ja+1) = r_boundary;% 奖励
        else % 向其他方向运动或保持
            tmp_pol(3) = 0;tmp_pol(4) = 0;%
            tmp_sar(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
            tmp_sar(2,ja+1) = pi_act(tmp_sar(1,ja+1),:)*markact;
            % 判断移动后的位置
            if (max(ismember(tmp_sar(1,ja+1),forb_s)) == 1 )% 禁区
                tmp_sar(3,ja+1) =  r_forbiden;
            elseif (max(ismember(tmp_sar(1,ja+1),targ_s)) == 1) % 目的地
                tmp_sar(3,ja+1) =  r_traget;
            else % 通行
                tmp_sar(3,ja+1) = r_pass;
            end
        end

    elseif ( cur_state == vert_ul )
        if ( tmp_pol(1) == 1 || tmp_pol(4) == 1) % 撞边界返回
            tmp_sar(1,ja+1) = cur_state;% 状态
            tmp_sar(2,ja+1) = pi_act(cur_state,:)*markact;% 行为
            tmp_sar(3,ja+1) = r_boundary;% 奖励
        else % 向其他方向运动或保持
            tmp_pol(1) = 0;tmp_pol(4) = 0;% 更新策略
            tmp_sar(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
            tmp_sar(2,ja+1) = pi_act(tmp_sar(1,ja+1),:)*markact;
            % 判断移动后的位置
            if (max(ismember(tmp_sar(1,ja+1),forb_s)) == 1 )% 禁区
                tmp_sar(3,ja+1) =  r_forbiden;
            elseif (max(ismember(tmp_sar(1,ja+1),targ_s)) == 1) % 目的地
                tmp_sar(3,ja+1) =  r_traget;
            else % 通行
                tmp_sar(3,ja+1) = r_pass;
            end
        end
    
    elseif ( cur_state == vert_ur )
        if ( tmp_pol(1) == 1 || tmp_pol(2) == 1) % 撞边界返回
            tmp_sar(1,ja+1) = cur_state;% 状态
            tmp_sar(2,ja+1) = pi_act(cur_state,:)*markact;% 行为
            tmp_sar(3,ja+1) = r_boundary;% 奖励
        else % 向其他方向运动或保持
            tmp_pol(1) = 0;tmp_pol(2) = 0;% 更新策略
            tmp_sar(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
            tmp_sar(2,ja+1) = pi_act(tmp_sar(1,ja+1),:)*markact;
            % 判断移动后的位置
            if (max(ismember(tmp_sar(1,ja+1),forb_s)) == 1 )% 禁区
                tmp_sar(3,ja+1) =  r_forbiden;
            elseif (max(ismember(tmp_sar(1,ja+1),targ_s)) == 1) % 目的地
                tmp_sar(3,ja+1) =  r_traget;
            else % 通行
                tmp_sar(3,ja+1) = r_pass;
            end
        end

    elseif ( cur_state == vert_lr )
        if ( tmp_pol(2) == 1 || tmp_pol(3) == 1) % 撞边界返回
            tmp_sar(1,ja+1) = cur_state;% 状态
            tmp_sar(2,ja+1) = pi_act(cur_state,:)*markact;% 行为
            tmp_sar(3,ja+1) = r_boundary;% 奖励
        else % 向其他方向运动或保持
            tmp_pol(2) = 0;tmp_pol(3) = 0;% 更新策略
            tmp_sar(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
            tmp_sar(2,ja+1) = pi_act(tmp_sar(1,ja+1),:)*markact;
            % 判断移动后的位置
            if (max(ismember(tmp_sar(1,ja+1),forb_s)) == 1 )% 禁区
                tmp_sar(3,ja+1) =  r_forbiden;
            elseif (max(ismember(tmp_sar(1,ja+1),targ_s)) == 1) % 目的地
                tmp_sar(3,ja+1) =  r_traget;
            else % 通行
                tmp_sar(3,ja+1) = r_pass;
            end
        end

    % 边界上的state
    elseif  ( max(ismember(cur_state, edge_l)) == 1 )
        if ( tmp_pol(4) == 1 ) % 撞边界返回
            tmp_sar(1,ja+1) = cur_state;% 状态
            tmp_sar(2,ja+1) = pi_act(cur_state,:)*markact;% 行为
            tmp_sar(3,ja+1) = r_boundary;% 奖励
        else % 向其他方向运动或保持
            tmp_pol(4) = 0;
            tmp_sar(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
            tmp_sar(2,ja+1) = pi_act(tmp_sar(1,ja+1),:)*markact;
            % 判断移动后的位置
            if (max(ismember(tmp_sar(1,ja+1),forb_s)) == 1 )% 禁区
                tmp_sar(3,ja+1) =  r_forbiden;
            elseif (max(ismember(tmp_sar(1,ja+1),targ_s)) == 1) % 目的地
                tmp_sar(3,ja+1) =  r_traget;
            else % 通行
                tmp_sar(3,ja+1) = r_pass;
            end
        end


    elseif  ( max(ismember(cur_state, edge_u)) == 1 )
        if ( tmp_pol(1) == 1 ) % 撞边界返回
            tmp_sar(1,ja+1) = cur_state;% 状态
            tmp_sar(2,ja+1) = pi_act(cur_state,:)*markact;% 行为
            tmp_sar(3,ja+1) = r_boundary;% 奖励
        else % 向其他方向运动或保持
            tmp_pol(1) = 0;
            tmp_sar(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
            tmp_sar(2,ja+1) = pi_act(tmp_sar(1,ja+1),:)*markact;
            % 判断移动后的位置
            if (max(ismember(tmp_sar(1,ja+1),forb_s)) == 1 )% 禁区
                tmp_sar(3,ja+1) =  r_forbiden;
            elseif (max(ismember(tmp_sar(1,ja+1),targ_s)) == 1) % 目的地
                tmp_sar(3,ja+1) =  r_traget;
            else % 通行
                tmp_sar(3,ja+1) = r_pass;
            end
        end

    elseif ( max(ismember(cur_state, edge_r)) == 1 )
        if ( tmp_pol(2) == 1 ) % 撞边界返回
            tmp_sar(1,ja+1) = cur_state;% 状态
            tmp_sar(2,ja+1) = pi_act(cur_state,:)*markact;% 行为
            tmp_sar(3,ja+1) = r_boundary;% 奖励
        else % 向其他方向运动或保持
            tmp_pol(2) = 0;
            tmp_sar(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
            tmp_sar(2,ja+1) = pi_act(tmp_sar(1,ja+1),:)*markact;
            % 判断移动后的位置
            if (max(ismember(tmp_sar(1,ja+1),forb_s)) == 1 )% 禁区
                tmp_sar(3,ja+1) =  r_forbiden;
            elseif (max(ismember(tmp_sar(1,ja+1),targ_s)) == 1) % 目的地
                tmp_sar(3,ja+1) =  r_traget;
            else % 通行
                tmp_sar(3,ja+1) = r_pass;
            end
        end

    elseif ( max(ismember(cur_state, edge_d)) == 1 )
        if ( tmp_pol(3) == 1 ) % 撞边界返回
            tmp_sar(1,ja+1) = cur_state;% 状态
            tmp_sar(2,ja+1) = pi_act(cur_state,:)*markact;% 行为
            tmp_sar(3,ja+1) = r_boundary;% 奖励
        else % 向其他方向运动或保持
            tmp_pol(3) = 0;
            tmp_sar(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
            tmp_sar(2,ja+1) = pi_act(tmp_sar(1,ja+1),:)*markact;
            % 判断移动后的位置
            if (max(ismember(tmp_sar(1,ja+1),forb_s)) == 1 )% 禁区
                tmp_sar(3,ja+1) =  r_forbiden;
            elseif (max(ismember(tmp_sar(1,ja+1),targ_s)) == 1) % 目的地
                tmp_sar(3,ja+1) =  r_traget;
            else % 通行
                tmp_sar(3,ja+1) = r_pass;
            end
        end

    %中间区域
    else
        tmp_sar(1,ja+1) = cur_state + vert_po*tmp_pol;% 更新位置
        tmp_sar(2,ja+1) = pi_act(tmp_sar(1,ja+1),:)*markact;
        % 判断移动后的位置
        if (max(ismember(tmp_sar(1,ja+1),forb_s)) == 1 )% 禁区
            tmp_sar(3,ja+1) =  r_forbiden;
        elseif (max(ismember(tmp_sar(1,ja+1),targ_s)) == 1) % 目的地
            tmp_sar(3,ja+1) =  r_traget;
        else % 通行
            tmp_sar(3,ja+1) = r_pass;
        end
    end
end
sta_act_rew = tmp_sar;% 返回n步的state，action and reward
end