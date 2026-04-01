% --------- TD , SARSA , n-steps sarsa , Q-Learning ---------------- %
%  - 模拟的初始条件 -
%  all the episodes start from the top-left state and terminate at the target state....
%  The reward settings are rtarget = 0, rforbidden = rboundary = −10, and rother = −1....
%  Moreover, αt(s,a) = 0.1 for all t and epsilon = 0.1....
%  The initial guesses of the action values are q0(s,a) = 0 for all (s,a)....
%  The initial policy has a uniform distribution: π0(a|s) = 0.2 for all s,a.
%% -------- sarsa and n-step sarsa and expected SARSA（期望）
% 1) 通过输入n-step来控制执行sarsa的类型
clear;clc;close all;

% - 生成初始策略（随机） - 
% 0:通行区域(白) 1:起点(黄) 2:终点(绿) 3:禁止区域(红) 
matrix_maze = [0 3 0 0 0;0 3 2 3 0;0 0 3 0 0;0 3 3 0 0;1 0 0 0 0];% 区域非点
color_map   = [1 1 1;1 1 0;0 1 0;1 0 0];                          % 各区域的颜色
arrow_mod   = [0.45,0.65;0.57,0.5;0.45,0.3;0.25,0.5;0.45,0.5];    % 绘制箭头的修正量
action2     = {'↑','→','↓','←','o'};% policy a1=1 a2=2 a3=3 a4=4 a5=5

% - 生成随机策略 - 
matrix_pi0 = randi(5, 5, 5);

% - policy and action value matrixs (初始概率均为0.2，探索为各向同性:Epsilon-Greedy)
epsilon     = 0.1;% \in [0 1]  between exploration(1) and exploitation(0) 
probability = [1-(length(action2)-1)/length(action2)*epsilon,epsilon/length(action2)];% 1最大概率；2其余概率
alpha       = 0.1;
pi_act      = 1/length(action2)*ones( length(matrix_pi0(:)) , length(action2));% 初始策略


% - reward and punishment - 
r_boundary = -10 ; r_forbiden = -10 ; r_traget = 2 ; r_pass = -0.2; gamma = 0.85;
% REW = {r_boundary,r_forbiden,r_traget,r_pass};

tmp_pi = reshape(matrix_pi0,[],1);
% - 网格的顶点和边界编号 - 
ng      = size(matrix_pi0,1);                                    % 迷宫中向左或向右跳的网格数
vert_ll = 1;vert_ul = ng;vert_ur = ng^2;vert_lr = ng^2-ng+1;     % 顶点
edge_l  = 2:ng-1;edge_u = ng*(2:ng-1);                           % 边界
edge_r  = (2:ng-1)+(ng-1)*ng;edge_d = 1+ng*(1:ng-2);             % 边界
forb_s  = find(matrix_maze == 3);targ_s = find(matrix_maze == 2);% 禁区
VERT_EDGE = {vert_ll,vert_ul,vert_ur,vert_lr,...
             edge_l,edge_u,edge_r,edge_d,forb_s,targ_s};         % 顶点；边界；禁区；目标（矩阵中个元素的顺序）


% - 输入n_setp 对应于 sarsa 的算法
n_setp = 10; % 例如：n = 1时是SARSA。

% - 迭代次数设置 - 
%  1）由起点到终点为一个episode，其长度不是固定点的，在达到目标后停止迭代
%  2）在每次使用使用sarsa（一小段pair）更新policy。
%  3）episode的个数（或者说迭代数）是固定的或者根据一些规则停止迭代,当前状态的标号是否等于目标的标号
n_epi = 500; % episode的个数（迭代数）
m_epi = (1+n_setp/20)*10^3; % 最大的episde长度

% - trajectory子程序设置 - 
vert_po = [1, ng, -1, -ng, 0];% state更新
markact = 1:5;                % action更新
% 为了避免频繁调用rand函数，一次性生成足够多的随机数(大于episode长度(未知))以便调用
nprob = rand(m_epi,1); % 随机数采样的概率
qvalue  = 0*pi_act;    % note:由于规则更倾向于扣分，因此将Q值初始值设为0，以鼓励更多地探索未访问过的状态-动作对.

% - 存储器(元数组)
cel_sta = {};
cel_act = {};
cel_rew = {};

% 生成n个episode
for j1 = 1:n_epi

    % - state & action & reward初始化
    tmp_sta = find( matrix_maze == 1 ); % 确定起始网格的位置
    tmp_act = 0;                        % 补全向量长度
    tmp_rew = 0;
    iter    = 1;

    %     qvalue  = 0*pi_act;

    % 停止迭代条件： 1)长度达到预定值后停止迭代
    while( iter <= m_epi )

        % Generate trajectory with policy
        % 由策略确定概率
        cur_state = tmp_sta(iter);%
        cumpol  = [0,cumsum(pi_act(cur_state, :), 2)]; % 在这里补个0
        ind_act = discretize(nprob(iter), cumpol);
        tmp_pol = zeros(5, 1);   % 列向量，便于点积和索引
        tmp_pol(ind_act) = 1;
        
        % 判断位置以根据polciy确定的动作以确定下一时刻的位置
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
            tmp_sta = [tmp_sta,cur_state];
            tmp_act = [tmp_act,markact * tmp_pol];
            tmp_rew = [tmp_rew,r_boundary];
        else
            % 正常移动
            tmp_sta = [tmp_sta,cur_state + vert_po * tmp_pol];
            tmp_act = [tmp_act,markact * tmp_pol];

            % 根据移动后的位置确定奖励
            if ismember(tmp_sta( iter + 1 ), forb_s)
                tmp_rew = [tmp_rew,r_forbiden];% 获得禁区惩罚
            elseif ismember(tmp_sta( iter + 1 ), targ_s)
                tmp_rew = [tmp_rew,r_traget];% 获得目标奖励
            else
                tmp_rew = [tmp_rew,r_pass];% 获得通行惩罚epsilon
            end
        end
        % - 优化计数策略
        iter = iter + 1;% 与tmp_sta的长度同步增长
        
        % - 根据输入的n-step的个数执行对应的n-steps sarsa
        % 关于n-step的sarsa方法的思考：
        %1)前n-step迭代不使用sarsa计算qvalue，保持待机，不进行policy的更新。
        %2)当iter<n-step时，执行的是iter-step的sarsa方法,主要是更新初始(s0,a1)的qvalue在之间时刻的值，并不用于更新其他(s,a)的qvalue。
        %  相反地，当iter>=n-step时，执行的是n-step的sarsa方法(可推进计算除初始时刻后续时刻的qvalue)
        %PS:前n-step次迭代之前更新初始状态的q值，但更新s1a2的时候q值是什么？是初始值，因此我们这里采用的方法似乎并不和谐。
        %3)执行sarsa的类型随着iter的增加改变，既第inter次迭代执行iter-step的sarsa。
        %  这种方法的问题在于只能计算第一时刻的q值，无法推进计算后续时刻的q值，如此便与MC方法的思想类似。
        if ( iter > n_setp + 1 ) % 这里主要使用的第1种方法更新q值，保证q值更新的统一性
            % 通过调整n-step实现不同SARSA的更新
            ind_row  = tmp_sta( iter - (n_setp + 1) );ind_col  = tmp_act( iter - n_setp );% n步之前S和A的index
            ind_row1 = tmp_sta( iter - 1 );ind_col1 = tmp_act( iter );% 最接近一次路线更新的索引
            
            sar_rew  = gamma.^(0:n_setp-1)*tmp_rew( iter - n_setp : iter - 1 )';
            tmp_qvalue = qvalue(ind_row,ind_col) - alpha*(qvalue(ind_row,ind_col) -...
                       (sar_rew + gamma^n_setp*qvalue(ind_row1,ind_col1)));
            qvalue(ind_row,ind_col) = tmp_qvalue;% 更新策略

            % Update policy (epsilon-greedy)
            % 找到最大的q对应动作，随机选一个最为最大概率(是否可以不适用find函数)
            ind_max = find( qvalue(ind_row,:) == max(qvalue(ind_row,:)) );
            ind     = randi(length(ind_max));

            pi_act( ind_row, 1:end ) = probability(2);
            pi_act( ind_row,ind_max(ind)) = probability(1);% 最大概率
        end

    end

end


% ------------------------------- 绘图 ---------------------------------- %
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
title([num2str(n_setp),'-step SARSA'],'FontSize', 20,'Fontname','times')
axis equal;axis off;drawnow;

%% ------------------------ Q-learning ----------------------------- %%
% on-policy
clear;clc;close all;

% - 生成初始策略（随机） - 
% 0:通行区域(白) 1:起点(黄) 2:终点(绿) 3:禁止区域(红) 
matrix_maze = [0 3 0 0 0;0 3 2 3 0;0 0 3 0 0;0 3 3 0 0;1 0 0 0 0];% 区域非点
color_map   = [1 1 1;1 1 0;0 1 0;1 0 0];                          % 各区域的颜色
arrow_mod   = [0.45,0.65;0.57,0.5;0.45,0.3;0.25,0.5;0.45,0.5];    % 绘制箭头的修正量
action2     = {'↑','→','↓','←','o'};% policy a1=1 a2=2 a3=3 a4=4 a5=5

% - 生成随机策略 - 
matrix_pi0 = randi(5, 5, 5);

% - policy and action value matrixs (初始概率均为0.2，探索为各向同性:Epsilon-Greedy)
epsilon     = 0.1;% \in [0 1]  between exploration(1) and exploitation(0) 
probability = [1-(length(action2)-1)/length(action2)*epsilon,epsilon/length(action2)];% 1最大概率；2其余概率
alpha       = 0.1;
pi_act      = 1/length(action2)*ones( length(matrix_pi0(:)) , length(action2));% 初始策略


% - reward and punishment - 
r_boundary = -10 ; r_forbiden = -10 ; r_traget = 1 ; r_pass = -0.05; gamma = 0.85;
% REW = {r_boundary,r_forbiden,r_traget,r_pass};

% - 网格的顶点和边界编号 - 
ng      = size(matrix_pi0,1);                                    % 迷宫中向左或向右跳的网格数
vert_ll = 1;vert_ul = ng;vert_ur = ng^2;vert_lr = ng^2-ng+1;     % 顶点
edge_l  = 2:ng-1;edge_u = ng*(2:ng-1);                           % 边界
edge_r  = (2:ng-1)+(ng-1)*ng;edge_d = 1+ng*(1:ng-2);             % 边界
forb_s  = find(matrix_maze == 3);targ_s = find(matrix_maze == 2);% 禁区
VERT_EDGE = {vert_ll,vert_ul,vert_ur,vert_lr,...
             edge_l,edge_u,edge_r,edge_d,forb_s,targ_s};         % 顶点；边界；禁区；目标（矩阵中个元素的顺序）

% - 迭代次数设置 - 
%  1）由起点到终点为一个episode，其长度不是固定点的，在达到目标后停止迭代
%  2）在每次使用使用sarsa（一小段pair）更新policy。
%  3）episode的个数（或者说迭代数）是固定的或者根据一些规则停止迭代,当前状态的标号是否等于目标的标号
n_epi = 500; % episode的个数（迭代数）
m_epi = 2*10^3; % 最大的episde长度

% - trajectory子程序设置 - 
vert_po = [1, ng, -1, -ng, 0];% state更新
markact = 1:5;                % action更新
% 为了避免频繁调用rand函数，一次性生成足够多的随机数(大于episode长度(未知))以便调用
nprob = rand(m_epi,1); % 随机数采样的概率
qvalue  = 0*pi_act;    % note:由于规则更倾向于扣分，因此将Q值初始值设为0，以鼓励更多地探索未访问过的状态-动作对.

% - 存储器(元数组)
cel_sta = {};
cel_act = {};
cel_rew = {};


for j1 = 1:n_epi

    % - state & action & reward初始化
    tmp_sta = find( matrix_maze == 1 ); % 确定起始网格的位置
    tmp_act = 0;                        % 补全向量长度
    tmp_rew = 0;
    iter    = 1;

    %     qvalue  = 0*pi_act;

    % 停止迭代条件： 1)长度达到预定值后停止迭代
    while( iter <= m_epi )

        % Generate trajectory with policy
        % 由策略确定概率
        cur_state = tmp_sta(iter);%
        cumpol  = [0,cumsum(pi_act(cur_state, :), 2)]; % 在这里补个0
        ind_act = discretize(nprob(iter), cumpol);
        tmp_pol = zeros(5, 1);   % 列向量，便于点积和索引
        tmp_pol(ind_act) = 1;
        
        % 判断位置以根据polciy确定的动作以确定下一时刻的位置
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
            tmp_sta = [tmp_sta,cur_state];
            tmp_act = [tmp_act,markact * tmp_pol];
            tmp_rew = [tmp_rew,r_boundary];
        else
            % 正常移动
            tmp_sta = [tmp_sta,cur_state + vert_po * tmp_pol];
            tmp_act = [tmp_act,markact * tmp_pol];

            % 根据移动后的位置确定奖励
            if ismember(tmp_sta( iter + 1 ), forb_s)
                tmp_rew = [tmp_rew,r_forbiden];% 获得禁区惩罚
            elseif ismember(tmp_sta( iter + 1 ), targ_s)
                tmp_rew = [tmp_rew,r_traget];% 获得目标奖励
            else
                tmp_rew = [tmp_rew,r_pass];% 获得通行惩罚epsilon
            end
        end
        % - 优化计数策略
        iter = iter + 1;% 与tmp_sta的长度同步增长

        % - q-learning 
        ind_row  = tmp_sta( iter - 1 );ind_col = tmp_act( iter );
        ind_row1 = tmp_sta( iter );
        
        % - 随机选取st+1的最大qvalue的值 - 
        indmax = find( qvalue(ind_row1,:) == max( qvalue(ind_row1,:) ) );
        indm   = randi( length(indmax) );

        ql_rew = tmp_rew(iter) + gamma*qvalue(ind_row1,indmax(indm));
        tmp_qvalue = qvalue(ind_row,ind_col) - alpha*(qvalue(ind_row,ind_col) - ql_rew);
        qvalue(ind_row,ind_col) = tmp_qvalue;% 更新策略

        % Update policy (epsilon-greedy)
        % 找到最大的q对应动作，随机选一个最为最大概率(是否可以不适用find函数)
        ind_max = find( qvalue(ind_row,:) == max(qvalue(ind_row,:)) );
        ind     = randi(length(ind_max));

        pi_act( ind_row, 1:end ) = probability(2);
        pi_act( ind_row,ind_max(ind)) = probability(1);% 最大概率

    end
end

% ------------------------------- 绘图 ---------------------------------- %
% - 绘制迷宫(标记的大小与概率相关) -
tmp_pi = reshape(matrix_pi0,[],1);
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
title('Q-liearning with on-policy','FontSize', 20,'Fontname','times')
axis equal;axis off;drawnow;


%% ------------------------ Q-learning ----------------------------- %%
% off-policy
clear;clc;close all;
% - 生成初始策略（随机） - 
% 0:通行区域(白) 1:起点(黄) 2:终点(绿) 3:禁止区域(红) 
matrix_maze = [0 3 0 0 0;0 3 2 3 0;0 0 3 0 0;0 3 3 0 0;1 0 0 0 0];% 区域非点
color_map   = [1 1 1;1 1 0;0 1 0;1 0 0];                          % 各区域的颜色
arrow_mod   = [0.45,0.65;0.57,0.5;0.45,0.3;0.25,0.5;0.45,0.5];    % 绘制箭头的修正量
action2     = {'↑','→','↓','←','o'};% policy a1=1 a2=2 a3=3 a4=4 a5=5

% - 生成随机策略 - 
matrix_pi0 = randi(5, 5, 5);

% - policy and action value matrixs (初始概率均为0.2，探索为各向同性:Epsilon-Greedy)
epsilon     = 0.1;% \in [0 1]  between exploration(1) and exploitation(0) 
probability = [1-(length(action2)-1)/length(action2)*epsilon,epsilon/length(action2)];% 1最大概率；2其余概率
alpha       = 0.1;
pi_act      = 1/length(action2)*ones( length(matrix_pi0(:)) , length(action2));% 初始策略
qvalue      = 0*pi_act;
% - reward and punishment - 
r_boundary = -10 ; r_forbiden = -10 ; r_traget = 1 ; r_pass = -0.05; gamma = 0.85;
REW = {r_boundary,r_forbiden,r_traget,r_pass};

% - 网格的顶点和边界编号 - 
ng      = size(matrix_pi0,1);                                    % 迷宫中向左或向右跳的网格数
vert_ll = 1;vert_ul = ng;vert_ur = ng^2;vert_lr = ng^2-ng+1;     % 顶点
edge_l  = 2:ng-1;edge_u = ng*(2:ng-1);                           % 边界
edge_r  = (2:ng-1)+(ng-1)*ng;edge_d = 1+ng*(1:ng-2);             % 边界
forb_s  = find(matrix_maze == 3);targ_s = find(matrix_maze == 2);% 禁区
VERT_EDGE = {vert_ll,vert_ul,vert_ur,vert_lr,...
             edge_l,edge_u,edge_r,edge_d,forb_s,targ_s};         % 顶点；边界；禁区；目标（矩阵中个元素的顺序）

% - 设置trajectory的长度
steps = 1e6;
mat_sta = find( matrix_maze == 1 ); % 初始化state
mat_act = 0;
mat_rew = 0;

% - generate a long episode with behiver policy - 
[sta_act_rew] = TRAJ(mat_sta(1), pi_act, VERT_EDGE, REW, steps, ng);

% - Q-learning with off-policy -
% 使用生成的trajectory来计算qvalue和updeta policy

for j1 = 1:length(sta_act_rew)-1
    
    ind_row  = sta_act_rew(1,j1);
    ind_col  = sta_act_rew(2,j1+1);
    ind_row1 = sta_act_rew(1,j1+1);

    % - 随机选取st+1的最大qvalue的值 -
    indmax = find( qvalue(ind_row1,:) == max( qvalue(ind_row1,:) ) );
    indm   = randi( length(indmax) );
    ql_rew = sta_act_rew(3,j1 + 1) + gamma*qvalue(ind_row1,indmax(indm));
    tmp_qvalue = qvalue(ind_row,ind_col) - alpha*(qvalue(ind_row,ind_col) - ql_rew);
    qvalue(ind_row,ind_col) = tmp_qvalue;% 更新策略
    
    % update policy with greedy 
    ind_max = find( qvalue(ind_row,:) == max(qvalue(ind_row,:)) );
    ind     = randi(length(ind_max));

    pi_act( ind_row, 1:end ) = probability(2);
    pi_act( ind_row,ind_max(ind)) = probability(1);% 最大概率
end

% ------------------------------- 绘图 ---------------------------------- %
% - 绘制迷宫(标记的大小与概率相关) -
tmp_pi = reshape(matrix_pi0,[],1);
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
title('Q-liearning with off-policy','FontSize', 20,'Fontname','times')
axis equal;axis off;drawnow;


%% ----------------------------- 子程序 --------------------------------- %%
% 由输入的policy和长度n，生成一个长度为n的trajectory
function [sta_act_rew] = TRAJ(state0, pi_act, VERT_EDGE, REW, steps, ng)
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
sta_act_rew = [tmp_sta; tmp_act; tmp_rew];
end