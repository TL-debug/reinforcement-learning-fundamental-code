% ----------- MC - epsilon greedy ------------------- %
% policy is stochastic 
% 问题由于policy是随机的，而执行action只能执行某一个action，...
% 那么怎么在选择其他的action以凸显出随机性。
% 建议使用随机数采样的方法来确定action；在确定过程中如何减少find的使用次数
% 使用 every-visit moethod 因为一个episode需要被充分利用
% 
% Note: 使用一个初始策略生成一个episode后（10^6），根据计算得到的qvalue，后更新策略
%       后续的episode的长度可根据需要减少
%% ---------------- MC epsilon-greedy ------------------- %%
% 根据预定的episode的长度，基于初始policy生成一段trajectory
% 由新trajectory计算q值，对最大q值赋值最大的概率，其余act小概率，以此更新policy
clear;clc;

% -生成初始策略（随机） - 
% 0:通行区域(白) 1:起点(黄) 2:终点(绿) 3:禁止区域(红) 
matrix_maze = [0 3 0 0 0;0 3 2 3 0;0 0 3 0 0;0 3 3 0 0;1 0 0 0 0];% 区域非点
color_map   = [1 1 1;1 1 0;0 1 0;1 0 0];                          % 各区域的颜色
arrow_mod   = [0.45,0.65;0.57,0.5;0.45,0.3;0.25,0.5;0.45,0.5];    % 绘制箭头的修正量
action2     = {'↑','→','↓','←','o'};% policy a1=1 a2=2 a3=3 a4=4 a5=5

% - 生成随机策略 - 
matrix_pi0 = randi(5, 5, 5);

% - policy and action value matrixs (初始概率均为0.2，探索为各向同性:Epsilon Greedy)
epsilon = 0.2;% \in [0 1]  between exploration(1) and exploitation(0) 
pi_act = 1/length(action2)*ones(size(matrix_pi0,1)*size(matrix_pi0,2),length(action2));


% - reward and punishment - 
r_boundary = -1 ; r_forbiden = -1 ; r_traget = 1 ; r_pass = 0; gamma = 0.9;
REW = {r_boundary,r_forbiden,r_traget,r_pass};

% - 布置迷宫 -
% 迷宫大小为5*5;形状为参考zhaoshiyu课程
% 禁止区域颜色为红色；通行区域白色；目标区域为绿色 (箭头的长短与概率大小相关)
tmp_pi = reshape(matrix_pi0,[],1);
for j2 = 1:length(tmp_pi)
    indx = floor(j2/(sqrt(length(tmp_pi))+0.1))+1;
    indy = j2 - sqrt(length(tmp_pi))*(indx -1);
    % 网格
    patch([indx indx+1 indx+1 indx],[indy indy indy+1 indy+1],...
          color_map(matrix_maze(indy,indx)+1,:),'EdgeColor','k','LineWidth',2);hold on;
    % 加速度(pi)
    for j3 = 1:size(pi_act,2)
        if ( pi_act(j2,j3) ~= 0 )
            text(indx+arrow_mod(j3,1),indy+arrow_mod(j3,2),action2(j3)...
                 ,'color',[0 0 0.8],'FontSize',100*pi_act(j2,j3));
        end
    end
end
axis equal;axis off;

% - 网格的顶点和边界编号 - 
ng = size(matrix_pi0,1);                                         % 迷宫中向左或向右跳的网格数
vert_ll = 1;vert_ul = ng;vert_ur = ng^2;vert_lr = ng^2-ng+1;     % 顶点
edge_l = 2:ng-1;edge_u = ng*(2:ng-1);                            % 边界
edge_r = (2:ng-1)+(ng-1)*ng;edge_d = 1+ng*(1:ng-2);              % 边界
forb_s = find(matrix_maze == 3);targ_s = find(matrix_maze == 2); % 禁区
VERT_EDGE = {vert_ll,vert_ul,vert_ur,vert_lr,...
             edge_l,edge_u,edge_r,edge_d,forb_s,targ_s};         % 顶点；边界；禁区；目标（矩阵中个元素的顺序）

% - episode设置 - 
% "独立的"按照随机数采样的方法和discretize函数实现执行action的选择 
%   epsilon确定后，概率的大小确定，最大概率的action需要根据q(s,a)确定
steps       = 1e6; % episode的长度
cum_pi_act  = cumsum(pi_act,2); % 列对应action标号；行对应的state
act_prob    = rand(length(pi_act),1);
iter        = 5;  % 总迭代数
probability = [1-(length(action2)-1)/length(action2)*epsilon,epsilon/length(action2)];% 1最大概率；2其余概率

mat_qva = zeros(size(matrix_maze,1)*size(matrix_maze,2),length(action2));% 内环完成之后计算
for jo = 1:iter % 使用内循环更新policy，state则使用初始值
    % - 存储器(向量)
    mat_sta = find( matrix_maze == 1 ); % 初始化state
    mat_act = 0;
    mat_rew = 0;
    

    % - 选择一个初始的state开始循环 -
    state0 = mat_sta(end);
    [sta_act_rew] = TRAJ_eg(state0, pi_act, VERT_EDGE, REW, steps, ng);

    % - 更新state action and reward -
    mat_sta = [mat_sta,sta_act_rew(1,:)];% 赋予初值以便计算
    mat_act = [mat_act,sta_act_rew(2,:)];
    mat_rew = [mat_rew,sta_act_rew(3,:)];
    % - policy evaluation（循环计算q值） -
    % 1)计算对应的q值加到对应的policy的位置；2)使用一个计数矩阵，记录访问次数
    % 3)q值矩阵与计数矩阵求平均，未访问的state和action的policy如何处理？
    % note:reward的使用是从最后一个到第二个，...
    %      然后根据action最后一个到第二个，state 导数第二到第一个的索引记录q和计数
    tmp_pol = 0*pi_act;% state 和 action 返回pi_act中的行与列
    tmp_cnt = 0*pi_act;% 统计对应state和action出现的次数
    iteq = length(mat_sta);
    for jq = 1:iteq - 1
        % - 索引位置
        row = mat_sta( iteq - jq );
        col = mat_act( iteq + 1 - jq );
        % - 计算累计q值
        if ( jq == 1 )
            tmpg = mat_rew( iteq + 1 - jq );
        else
            tmpg = mat_rew( iteq + 1 - jq ) + gamma*tmpg;
        end
        tmp_pol(row,col) = tmp_pol(row,col) + tmpg;
        tmp_cnt(row,col) = tmp_cnt(row,col) + 1;
    end

    % - policy improvement - (由返回的qvalue更新policy，矩阵当中存在NaN)
    % 1) 鼓励探索未知区域：q值初始值为0，惩罚的q值为负数，在policy中未探索的action随机选取一个给最大的概率
    % 2) 鼓励探索已知区域：未探索的区域为NaN，在policy中，对已探索的行为下选择q值最大的act赋值最大概率
    tmp_mat_qva = tmp_pol./tmp_cnt; % 计算平均的q值
    tmp_mat_qva(isnan(tmp_mat_qva)) = 0;

    % - 找到探索位置，对policy进行更新
    indx = find( all( tmp_mat_qva == 0 , 2 ) == 0 );% 对indx = 0对应的行进行更新策略
    % - 更新policy（若网格世界很大？）-
    for jj = 1:length(indx)
        % 已探索state找到action的qvalue的最大值，赋值最大的概率，其余相等的小概率
        max_qvalue = find( tmp_mat_qva(indx(jj),:) == max(tmp_mat_qva(indx(jj),:)) );
        idx = randi(length(max_qvalue));
        pi_act(indx(jj),max_qvalue(idx)) = probability(1);% 最大概率
        pi_act(indx(jj),1:end ~= max_qvalue(idx) ) = probability(2);
    end

    % - 绘制迷宫(标记的大小与概率相关) -
    for j2 = 1:length(tmp_pi)
        indx = floor(j2/(sqrt(length(tmp_pi))+0.1))+1;
        indy = j2 - sqrt(length(tmp_pi))*(indx -1);
        % 网格
        patch([indx indx+1 indx+1 indx],[indy indy indy+1 indy+1],...
            color_map(matrix_maze(indy,indx)+1,:),'EdgeColor','k','LineWidth',2);hold on;
        % 加速度(pi)
        for j3 = 1:size(pi_act,2)
            % 根据概率选择箭头标记的大小
            if ( pi_act(j2,j3) == probability(1) )
                a = 0.8;
            else
                a = 0.4;
            end
            if ( pi_act(j2,j3) ~= 0 )
                text(indx+arrow_mod(j3,1),indy+arrow_mod(j3,2),action2(j3)...
                    ,'color',[0 0.0 0.6],'FontSize',30*a);
            end
        end
    end
    title(['iteration = ',num2str(jo)],'Fontsize',30)
    axis equal;axis off;drawnow;
end
%% ----------- 子程序 ------------
function [sta_act_rew] = TRAJ_eg(state0, pi_act, VERT_EDGE, REW, steps, ng) % steps为生成trajectory的长度
% 初始状态、策略、边界/禁区/目标位置、奖励、步长、网格尺寸

% 提取边界和特殊位置
vert_ll = cell2mat(VERT_EDGE(1)); vert_ul = cell2mat(VERT_EDGE(2));
vert_ur = cell2mat(VERT_EDGE(3)); vert_lr = cell2mat(VERT_EDGE(4));
edge_l  = cell2mat(VERT_EDGE(5)); edge_u  = cell2mat(VERT_EDGE(6));
edge_r  = cell2mat(VERT_EDGE(7)); edge_d  = cell2mat(VERT_EDGE(8));
forb_s  = cell2mat(VERT_EDGE(9)); targ_s  = cell2mat(VERT_EDGE(10));

% 提取奖励值
r_boundary = cell2mat(REW(1)); r_forbiden = cell2mat(REW(2));
r_traget   = cell2mat(REW(3)); r_pass     = cell2mat(REW(4));

% 动作对应的位移向量 (↑ → ↓ ← ○)
vert_po = [1, ng, -1, -ng, 0];

% 随机数序列
nprob = rand(steps, 1);
markact = 1:5;

% 预分配记忆数组（长度 steps+1）
tmp_sta = zeros(1, steps+1);
tmp_act = zeros(1, steps+1);
tmp_rew = zeros(1, steps+1);
tmp_sta(1) = state0;

for ja = 1:steps
    cur_state = tmp_sta(ja);

    % 根据策略确定动作
    cumpol  = [0,cumsum(pi_act(cur_state, :), 2)]; % 在这里补个0
    ind_act = discretize(nprob(ja), cumpol);
    tmp_pol = zeros(size(pi_act, 2), 1);   % 列向量，便于点积和索引
    tmp_pol(ind_act) = 1;

    % 判断是否撞墙（根据当前状态类型）
    collision = false;
    if cur_state == vert_ll
        if tmp_pol(3)==1 || tmp_pol(4)==1; collision = true; end
    elseif cur_state == vert_ul
        if tmp_pol(1)==1 || tmp_pol(4)==1; collision = true; end
    elseif cur_state == vert_ur
        if tmp_pol(1)==1 || tmp_pol(2)==1; collision = true; end
    elseif cur_state == vert_lr
        if tmp_pol(2)==1 || tmp_pol(3)==1; collision = true; end
    elseif any(ismember(cur_state, edge_l))
        if tmp_pol(4)==1; collision = true; end
    elseif any(ismember(cur_state, edge_u))
        if tmp_pol(1)==1; collision = true; end
    elseif any(ismember(cur_state, edge_r))
        if tmp_pol(2)==1; collision = true; end
    elseif any(ismember(cur_state, edge_d))
        if tmp_pol(3)==1; collision = true; end
    end

    if collision
        % 撞墙：状态不变，获得边界惩罚
        tmp_sta(ja+1) = cur_state;
        tmp_act(ja+1) = markact * tmp_pol;   % 点积得到动作编号
        tmp_rew(ja+1) = r_boundary;
    else
        % 正常移动
        tmp_sta(ja+1) = cur_state + vert_po * tmp_pol;
        tmp_act(ja+1) = markact * tmp_pol;

        % 根据移动后的位置确定奖励
        if ismember(tmp_sta(ja+1), forb_s)
            tmp_rew(ja+1) = r_forbiden;
        elseif ismember(tmp_sta(ja+1), targ_s)
            tmp_rew(ja+1) = r_traget;
        else
            tmp_rew(ja+1) = r_pass;
        end
    end
end

% 输出结果（从第二步开始）
sta_act_rew = [tmp_sta(2:end); tmp_act(2:end); tmp_rew(2:end)];
end