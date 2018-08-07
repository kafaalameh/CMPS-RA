clear all
close all
clc
set(0,'DefaultFigureWindowStyle','docked')

% A = load('arc130_SVD.mat');
% M = A.problem;
% spy(M);

A = load('arc130.mtx');
M = sparse(A(:,1),A(:,2),A(:,3));
spy(M);