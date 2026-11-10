# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.8.0", "numpy==1.26.4", "crafter==1.8.3", "gym==0.26.2",
#   "opensimplex==0.3", "pillow==11.3.0", "matplotlib==3.10.6",
#   "imageio==2.37.0", "imageio-ffmpeg==0.6.0",
# ]
# [tool.uv]
# extra-build-variables = { crafter = { PYTHONUTF8 = "1" }, gym = { PYTHONUTF8 = "1" } }
# ///
"""Crafter: código independente extraído de crafter_meta_goals.ipynb e adaptado.

RL² com PPO recorrente, sem professor, imitação, rótulo do objetivo ou replay.
A GRU recebe observação, ação/recompensa/done anteriores. Sua memória atravessa
os K episódios (padrão 3) com o mesmo objetivo oculto e zera só ao iniciar outra tarefa.
O PPO atualiza os pesos entre lotes de tarefas; a adaptação dentro de uma tarefa
acontece na memória, sem atualização de pesos.

Execução com dependências automáticas (requer uv e acesso à internet):
    uv run --python 3.12 crafter_meta_goals_rl2_ppo.py --smoke
    uv run --python 3.12 crafter_meta_goals_rl2_ppo.py --video
Também funciona com python 3.10–3.12 e as dependências acima já instaladas.
Use --help para parâmetros, --resume last.pt para retomar e
--eval-only --checkpoint best.pt para avaliar (mesmo --hidden do treino).
Checkpoints do notebook original são incompatíveis com esta arquitetura.
O teste smoke verifica funcionamento; não demonstra aprendizado/convergência.
As quatro categorias de objetivo são compartilhadas entre treino/validação/teste;
sementes de mundos são separadas. Não é teste de categorias de objetivo inéditas.
Mantém a observação semântica local e o ciclo diurno modificado do notebook;
não equivale ao benchmark Crafter padrão baseado em pixels.

Referências:
    RL²: https://arxiv.org/abs/1611.02779
    PPO: https://arxiv.org/abs/1707.06347
Notebook de origem SHA256: ba9a770e0089e6265356024fa5f5d0932efc7025934bbd6be0a867abb7c851a5

Mudanças em relação à v1 (com os valores padrão, o comportamento é o da v1):
    Recompensa:
        --reward-cap N     no máximo N eventos do objetivo recompensados por episódio
                           (0 = sem teto, como na v1; 1 = só o primeiro evento).
        --wrong-penalty P  -P a cada evento de um dos outros 3 objetivos (padrão 0).
        --reward eventos|primeiro  atalhos para --reward-cap 0 e 1.
        Na v1, beber água rende dezenas de +1 por episódio; "sempre beber" vira uma
        estratégia fixa quase ótima e quase não há ganho em identificar o objetivo.
    --episodes K   episódios por trial (padrão 3).
    --no-memory    controle: a GRU recebe estado zerado a cada passo. Mesma arquitetura
                   e mesmas entradas (inclusive a recompensa anterior), mas sem memória.
    O Crafter deixa de desenhar a imagem 64x64 a cada passo (o agente usa só o mapa
        semântico): o ambiente fica ~1,6x mais rápido. O jogo não muda: o desenho só
        sorteia ruído à noite (luz < 0,5), o que não acontece em horizontes de até ~700
        passos com o dia 5x mais lento; v1 e v2 geram as mesmas observações e recompensas.
    Avaliar (--eval-only) com recompensa, episódios ou memória diferentes do treino é recusado.
    A validação usa no máximo 4 ambientes (as 4 tarefas), sem rodar trials de enchimento.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

DOMAIN = 'crafter'
VERSION = 'crafter-rl2-ppo-v3'
K = 3
OBJETIVOS = ['collect_wood', 'collect_drink', 'eat_cow', 'collect_sapling']
NA = 6
AGUA, GRAMA, CAMINHO, AREIA, ARVORE, VACA = 1, 2, 4, 5, 6, 14
ANDAVEL = {GRAMA, CAMINHO, AREIA}
MOV = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
OBS_DIM = 45 + 63 * 8
INIMIGOS, OBSTACULOS = {15, 16, 17}, {3, 7, 8, 9, 10, 11, 12}
TETO_RECOMPENSA = 0      # máximo de eventos recompensados por episódio; 0 = sem teto (definido em main)
PENALIDADE_ERRADO = 0.0  # recompensa negativa por evento de outro objetivo (definida em main)

class Jogo:
    """Um mundo Crafter com recompensa oculta (+1 por evento real do objetivo). Observação = características da janela 9x7 + inventário/direção/luz."""

    def __init__(self):
        import crafter
        from crafter import constants

        class Dia(crafter.Env):

            def render(self, size=None):
                # step()/reset() desenham uma imagem 64x64 a cada passo, mas o agente usa
                # só o mapa semântico. Pular o desenho deixa o jogo ~1,6x mais rápido.
                # O vídeo continua funcionando: ele chama render com tamanho.
                if size is None:
                    return None
                return super().render(size)

            def _update_time(self):
                p = self._step / 1500 % 1 + 0.3
                self._world.daylight = 1 - np.abs(np.cos(np.pi * p)) ** 3

        class StableDia(Dia):

            def _balance_object(self, chunk, objs, *args, **kwargs):
                ordered = sorted(objs, key=lambda obj: tuple((int(x) for x in obj.pos)))
                return super()._balance_object(chunk, ordered, *args, **kwargs)
        self.cls, self.itens = (StableDia, list(constants.items.keys()))
        self.env = None

    def _obs(self):
        """Vetor de 549 números calculado SÓ a partir da janela 9x7 que o jogador vê (sem mapa global)."""
        e, p = (self.env, self.env._player)
        sem = np.pad(e._sem_view(), 4, constant_values=self.pad)
        x, y = (int(p.pos[0]) + 4, int(p.pos[1]) + 4)
        g = self.g2 = sem[x - 4:x + 5, y - 3:y + 4].astype(np.int64)
        f = []
        for tipos in ({ARVORE}, {AGUA}, {VACA}, {GRAMA}):
            melhor = None
            for i in range(9):
                for j in range(7):
                    if (i, j) != (4, 3) and g[i, j] in tipos:
                        d = abs(i - 4) + abs(j - 3)
                        if melhor is None or d < melhor[0]:
                            melhor = (d, i - 4, j - 3)
            f += [1.0, melhor[1] / 4, melhor[2] / 3] if melhor else [0.0, 0.0, 0.0]
        fx, fy = (int(v) for v in p.facing)
        t = g[4 + fx, 3 + fy]
        cat = 0 if t == GRAMA else 1 if t in (CAMINHO, AREIA) else 2 if t == AGUA else 3 if t == ARVORE else 4 if t == VACA else 5 if t in INIMIGOS else 6 if t in OBSTACULOS else 7
        frente = np.zeros(8)
        frente[cat] = 1
        livres = [float(g[4 + dx, 3 + dy] in ANDAVEL) for dx, dy in MOV.values()]
        inv = np.array([p.inventory[k] for k in self.itens], np.float32) / 9.0
        dir_ = np.zeros(4, np.float32)
        dir_[{(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}.get((fx, fy), 0)] = 1
        cat_grid = np.full(g.shape, 7, dtype=int)
        for idx, values in enumerate(({GRAMA}, {CAMINHO, AREIA}, {AGUA}, {ARVORE}, {VACA}, INIMIGOS, OBSTACULOS)):
            cat_grid[np.isin(g, list(values))] = idx
        grid = np.eye(8, dtype=np.float32)[cat_grid].ravel()
        return np.concatenate([f, frente, livres, inv, dir_, [e._world.daylight], grid]).astype(np.float32)

    def reset(self, seed, objetivo):
        self.env = self.cls(seed=int(seed), length=None)
        self.env.reset()
        self.pad = int(max(self.env._world._mat_ids.values()) + 7)
        self.obj, self.prev, self.morto = (OBJETIVOS[objetivo], 0, False)
        self.prev_errados = 0
        self.rng, self.cnt, self.dir = (np.random.default_rng(int(seed) % 2 ** 31), 0, 1)
        o = self._obs()
        return (o,)

    def step(self, a):
        """-> observação, recompensa, morreu, contagem_do_objetivo"""
        p = self.env._player
        if not self.morto:
            self.env.step(int(a))
            self.morto = p.health <= 0
        n = p.achievements.get(self.obj, 0)
        o = self._obs()
        # Os contadores de conquistas zeram a cada episódio (mundo novo).
        if TETO_RECOMPENSA > 0:
            r = float(min(n, TETO_RECOMPENSA) - min(self.prev, TETO_RECOMPENSA))
        else:
            r = float(n - self.prev)
        if PENALIDADE_ERRADO > 0:
            errados = sum(p.achievements.get(nome, 0) for nome in OBJETIVOS if nome != self.obj)
            r -= PENALIDADE_ERRADO * (errados - self.prev_errados)
            self.prev_errados = errados
        self.prev = n
        return (o, r, self.morto, n)

    def render(self, tam=320):
        state = self.env._world.random.get_state()
        try:
            return self.env.render((tam, tam))
        finally:
            self.env._world.random.set_state(state)


class TaskEnv:
    def __init__(self, horizon):
        self.game = Jogo()
        self.horizon = horizon

    def reset(self, target, seed):
        self.steps = 0
        (obs,) = self.game.reset(seed, target)
        if obs.shape != (OBS_DIM,):
            raise ValueError(f'Observação incompatível: {obs.shape}; esperado {(OBS_DIM,)}')
        return obs

    def step(self, action):
        obs, reward, dead, count = self.game.step(action)
        self.steps += 1
        return obs, reward, dead or self.steps >= self.horizon, {'success': count > 0}

    def render(self):
        return self.game.render()

    def close(self):
        self.game.env = None


class Vetor:
    def __init__(self, count, horizon):
        self.envs = [TaskEnv(horizon) for _ in range(count)]
        self.n, self.horizon = count, horizon

    def close(self):
        for env in self.envs:
            env.close()


class Agente(nn.Module):
    def __init__(self, hidden=128, sem_memoria=False):
        super().__init__()
        self.hid, self.actions = hidden, NA
        self.sem_memoria = sem_memoria  # controle: estado zerado a cada passo
        self.enc = nn.Sequential(nn.Linear(OBS_DIM + 2, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden + NA + 2, hidden)
        self.actor = nn.Linear(hidden, NA)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs, context, hidden, active=None):
        # [tempo, tarefas, atributos]. Nunca embaralhar passos recorrentes.
        encoded = torch.cat([self.enc(obs), context], dim=-1)
        if active is None and not self.sem_memoria:
            z, hidden = self.gru(encoded, hidden)
        else:
            states = []
            for t in range(encoded.shape[0]):
                anterior = torch.zeros_like(hidden) if self.sem_memoria else hidden
                _, candidate = self.gru(encoded[t:t + 1], anterior)
                if active is None:
                    hidden = candidate
                else:
                    valid = active[t][None, :, None]
                    hidden = torch.where(valid, candidate, hidden)
                states.append(hidden[0])
            z = torch.stack(states)
        return self.actor(z), self.critic(z).squeeze(-1), hidden


def sample_targets(rng, count):
    # Objetivo independente da semente de mundo; identidade não entra na rede.
    return rng.integers(len(OBJETIVOS), size=count).tolist()


@torch.no_grad()
def collect(vector, model, targets, seed, *, reset_memory=False,
            random_policy=False, record=False, deterministic=False):
    device = next(model.parameters()).device
    batch, horizon = vector.n, vector.horizon
    hidden = torch.zeros(1, batch, model.hid, device=device)
    context = torch.zeros(batch, NA + 2, device=device)
    data = {key: [] for key in ('obs', 'context', 'active', 'actions', 'logp', 'values', 'rewards')}
    rows, frames = [], [[] for _ in range(K)]
    for episode in range(K):
        if episode and reset_memory:
            hidden.zero_()
            context.zero_()
        observed = [env.reset(target, int(seed + episode * 1009 + b * 100003))
                    for b, (env, target) in enumerate(zip(vector.envs, targets))]
        alive = np.ones(batch, dtype=bool)
        total, success = np.zeros(batch), np.zeros(batch, dtype=bool)
        first = np.full(batch, horizon, dtype=int)
        for t in range(horizon):
            obs = torch.as_tensor(np.c_[np.stack(observed), np.full(batch, t / horizon),
                                       np.full(batch, episode / K)], dtype=torch.float32, device=device)
            active = torch.as_tensor(alive.copy(), device=device)
            logits, values, hidden = model(obs[None], context[None], hidden, active[None])
            distribution = Categorical(logits=logits[0])
            action = distribution.probs.argmax(-1) if deterministic else distribution.sample()
            if random_policy:
                action = torch.randint(NA, (batch,), device=device)
            data['obs'].append(obs.cpu())
            data['context'].append(context.cpu().clone())
            data['active'].append(active.cpu())
            data['actions'].append(action.cpu())
            data['logp'].append(distribution.log_prob(action).cpu())
            data['values'].append(values[0].cpu())
            rewards, done = np.zeros(batch, np.float32), np.zeros(batch, bool)
            for b, (env, a) in enumerate(zip(vector.envs, action.cpu().tolist())):
                if not alive[b]:
                    continue  # Não avançar mundos terminados; padding não altera memória.
                observed[b], rewards[b], done[b], info = env.step(a)
                total[b] += rewards[b]
                if info['success'] and not success[b]:
                    first[b] = t + 1
                success[b] |= info['success']
            data['rewards'].append(torch.from_numpy(rewards.copy()))
            encoded_action = nn.functional.one_hot(action, NA).float()
            boundary = torch.as_tensor(done | (t == horizon - 1), dtype=torch.float32, device=device)
            next_context = torch.cat([encoded_action,
                torch.as_tensor(rewards, device=device)[:, None], boundary[:, None]], dim=-1)
            context = torch.where(active[:, None], next_context, context)
            alive &= ~done
            if record:
                frames[episode].append(vector.envs[0].render())
        for b in range(batch):
            rows.append(dict(episode=episode + 1, task=b, return_=float(total[b]),
                             success=int(success[b]), first_success=int(first[b])))
    return {key: torch.stack(value) for key, value in data.items()}, rows, frames


def advantages(data, gamma, gae_lambda):
    # Episódios internos não encerram o retorno da tarefa RL². Só o fim do trial
    # (3 episódios) é terminal. Padding de mortes é ignorado, inclusive no desconto.
    rewards, values, active = data['rewards'], data['values'], data['active']
    adv = torch.zeros_like(rewards)
    following_value = torch.zeros_like(values[0])
    following_adv = torch.zeros_like(values[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * following_value - values[t]
        candidate = delta + gamma * gae_lambda * following_adv
        adv[t] = torch.where(active[t], candidate, 0.0)
        following_value = torch.where(active[t], values[t], following_value)
        following_adv = torch.where(active[t], candidate, following_adv)
    returns = adv + values
    selected = adv[active]
    normalized = (adv - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-8)
    return torch.where(active, normalized, 0.0), returns


def ppo_update(model, optimizer, data, cfg):
    adv, returns = advantages(data, cfg['gamma'], cfg['gae_lambda'])
    device = next(model.parameters()).device
    d = {key: value.to(device) for key, value in data.items()}
    adv, returns = adv.to(device), returns.to(device)
    mask = d['active']
    metrics = {}
    # Um lote contém trials completos, sempre da política atual. Reconstitui toda
    # a GRU a cada época, com backprop pelos três episódios; sem replay/burn-in.
    for epoch in range(cfg['ppo_epochs']):
        logits, values, _ = model(d['obs'], d['context'],
            torch.zeros(1, mask.shape[1], model.hid, device=device), mask)
        dist = Categorical(logits=logits)
        log_ratio = dist.log_prob(d['actions'])[mask] - d['logp'][mask]
        ratio = log_ratio.exp()
        approximate_kl = ((ratio - 1) - log_ratio).mean()
        if not torch.isfinite(approximate_kl):
            raise FloatingPointError('KL não finita; checkpoint anterior preservado.')
        if epoch and approximate_kl.item() > cfg['target_kl']:
            break
        objective = ratio * adv[mask]
        clipped = ratio.clamp(1 - cfg['clip'], 1 + cfg['clip']) * adv[mask]
        policy_loss = -torch.minimum(objective, clipped).mean()
        clipped_value = d['values'] + (values - d['values']).clamp(-cfg['clip'], cfg['clip'])
        value_loss = 0.5 * torch.maximum((values - returns).square(),
                                        (clipped_value - returns).square())[mask].mean()
        entropy = dist.entropy()[mask].mean()
        loss = policy_loss + cfg['value_coef'] * value_loss - cfg['entropy_coef'] * entropy
        if not torch.isfinite(loss):
            raise FloatingPointError('Perda não finita; checkpoint anterior preservado.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
        if not torch.isfinite(grad_norm):
            raise FloatingPointError('Gradiente não finito; checkpoint anterior preservado.')
        optimizer.step()
        metrics = dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
            entropy=float(entropy.detach()), approximate_kl=float(approximate_kl.detach()),
            clip_fraction=float(((ratio - 1).abs() > cfg['clip']).float().mean().detach()),
            ppo_epochs=epoch + 1, gradient_norm=float(grad_norm))
    return metrics


def atomic_save(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def training_contract(cfg):
    keys = ('iterations', 'horizon', 'hidden', 'seed', 'envs', 'ppo_epochs', 'gamma',
            'gae_lambda', 'clip', 'lr', 'entropy_coef', 'value_coef', 'max_grad_norm',
            'target_kl', 'validate_every', 'device', 'reward_cap', 'wrong_penalty', 'no_memory')
    return dict(domain=DOMAIN, episodes_per_trial=K, **{key: cfg[key] for key in keys})


def check_checkpoint(state, cfg, resume=False):
    if state.get('version') != VERSION:
        raise ValueError('Checkpoint incompatível: use pesos desta versão RL²/PPO.')
    contract = state['contract']
    if contract['domain'] != DOMAIN or contract['hidden'] != cfg['hidden']:
        raise ValueError('Arquitetura incompatível; preserve --hidden do treino.')
    # A recompensa anterior entra na GRU e o relógio de episódio depende de K: avaliar com
    # outra configuração mudaria o comportamento do agente sem aviso.
    for key, padrao in (('reward_cap', 0), ('wrong_penalty', 0.0), ('no_memory', False)):
        if contract.get(key, padrao) != cfg[key]:
            raise ValueError(f'Checkpoint treinado com {key}={contract.get(key, padrao)}; '
                             'avalie com a mesma recompensa e o mesmo modo de memória.')
    if contract.get('episodes_per_trial', 3) != cfg['episodes']:
        raise ValueError(f"Checkpoint treinado com --episodes {contract.get('episodes_per_trial', 3)}.")
    if resume and contract != training_contract(cfg):
        raise ValueError('Para retomar, preserve os parâmetros de treino registrados em config.json.')


def validation_score(vector, model):
    rows = []
    targets = list(range(len(OBJETIVOS)))
    # Validation worlds have a disjoint seed range from train and test.
    with torch.random.fork_rng(devices=[next(model.parameters()).device] if next(model.parameters()).is_cuda else []):
        torch.manual_seed(777)
        for lo in range(0, len(targets), vector.n):
            batch = targets[lo:lo + vector.n]
            real = len(batch)
            batch += [batch[0]] * (vector.n - real)
            _, part, _ = collect(vector, model, batch, 1_300_000_000 + lo * 100003)
            rows.extend(r for r in part if r['task'] < real and r['episode'] == K)
    return (float(np.mean([r['success'] for r in rows])),
            float(np.mean([r['return_'] for r in rows])))


def train(cfg, output):
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    model = Agente(cfg['hidden'], sem_memoria=cfg['no_memory']).to(cfg['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], eps=1e-5)
    history, start_iteration = [], 0
    best_score = (-float('inf'), -float('inf'))
    best_model = copy.deepcopy(model.state_dict())
    if cfg['resume']:
        saved = torch.load(cfg['resume'], map_location='cpu', weights_only=True)
        check_checkpoint(saved, cfg, resume=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        history, start_iteration = saved['history'], saved['iteration'] + 1
        best_score, best_model = saved['best_score'], saved['best_model']
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        if cfg['device'] == 'cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        atomic_save(output / 'best.pt', dict(version=VERSION, model=best_model,
                                            contract=training_contract(cfg)))
    vector = Vetor(cfg['envs'], cfg['horizon'])
    validation = Vetor(min(cfg['envs'], len(OBJETIVOS)), cfg['horizon'])
    started = time.monotonic()
    try:
        for it in range(start_iteration, cfg['iterations']):
            if time.monotonic() - started >= cfg['max_minutes'] * 60:
                print('Orçamento de treino atingido; retome usando last.pt.', flush=True)
                break
            model.train()
            for group in optimizer.param_groups:
                group['lr'] = cfg['lr'] * (1 - it / cfg['iterations'])
            targets = sample_targets(rng, vector.n)
            # Reserve disjoint numeric seed ranges even with padding and K resets.
            data, rows, _ = collect(vector, model, targets, int(rng.integers(1, 100_000_000)))
            row = dict(iteration=it + 1, train_success=float(np.mean([r['success'] for r in rows])),
                       train_return=float(np.mean([r['return_'] for r in rows])),
                       transitions=int(data['active'].sum()), **ppo_update(model, optimizer, data, cfg))
            if it % cfg['validate_every'] == 0 or it == cfg['iterations'] - 1:
                model.eval()
                score = validation_score(validation, model)
                row.update(validation_success=score[0], validation_return=score[1])
                if score > best_score:
                    best_score, best_model = score, copy.deepcopy(model.state_dict())
                    atomic_save(output / 'best.pt', dict(version=VERSION, model=best_model,
                                                        contract=training_contract(cfg)))
            row['seconds'] = time.monotonic() - started
            history.append(row)
            atomic_save(output / 'last.pt', dict(version=VERSION, model=model.state_dict(),
                optimizer=optimizer.state_dict(), history=history, iteration=it,
                best_score=best_score, best_model=best_model, contract=training_contract(cfg),
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if cfg['device'] == 'cuda' else None))
            (output / 'training.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
    finally:
        vector.close()
        validation.close()
    if not (output / 'best.pt').exists():
        raise RuntimeError('Nenhuma atualização/validação completou; verifique orçamento de tempo.')
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=True)['model'])
    cfg['completed_updates'] = len(history)
    return model


def evaluate(cfg, model, output):
    model = copy.deepcopy(model).cpu().eval()
    targets = list(range(len(OBJETIVOS))) * cfg['test_repeats']
    all_rows, modes = [], ('memory', 'reset', 'untrained', 'random')
    vector = Vetor(cfg['envs'], cfg['horizon'])
    try:
        for mode in modes:
            torch.manual_seed(12345)
            current = (Agente(cfg['hidden'], sem_memoria=cfg['no_memory']).eval()
                       if mode == 'untrained' else model)
            # Use the same action-sampling seed and worlds across paired conditions.
            torch.manual_seed(54321)
            for lo in range(0, len(targets), vector.n):
                batch = targets[lo:lo + vector.n]
                real = len(batch)
                batch += [batch[0]] * (vector.n - real)
                _, rows, _ = collect(vector, current, batch, 1_600_000_000 + lo * 100003,
                    reset_memory=mode == 'reset', random_policy=mode == 'random')
                for row in rows:
                    if row['task'] < real:
                        row.update(mode=mode, target=OBJETIVOS[batch[row['task']]], task_id=lo + row['task'])
                        all_rows.append(row)
            print('Avaliação concluída:', mode, flush=True)
    finally:
        vector.close()
    with (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = []
    for mode in modes:
        for ep in range(1, K + 1):
            subset = [r for r in all_rows if r['mode'] == mode and r['episode'] == ep]
            summary.append(dict(mode=mode, episode=ep, n=len(subset),
                success=float(np.mean([r['success'] for r in subset])),
                return_=float(np.mean([r['return_'] for r in subset])),
                time_to_success_capped=float(np.mean([r['first_success'] for r in subset]))))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for mode in modes:
        subset = [r for r in summary if r['mode'] == mode]
        for ax, metric, label in zip(axes, ('success', 'return_', 'time_to_success_capped'),
                                    ('Sucesso real', 'Recompensa', 'Passos (falha = horizonte)')):
            ax.plot(range(1, K + 1), [r[metric] for r in subset], marker='o', label=mode)
            ax.set_title(label)
            ax.set_xlabel('Episódio da mesma tarefa')
            ax.set_xticks(range(1, K + 1))
            ax.grid(alpha=0.3)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output / 'adaptacao.png', dpi=130)
    plt.close(figure)
    if cfg['video']:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        vector = Vetor(1, cfg['horizon'])
        try:
            torch.manual_seed(54321)
            with imageio.get_writer(output / 'demonstracao.mp4', fps=20,
                                    codec='libx264', macro_block_size=2) as movie:
                for index, target in enumerate(targets[:cfg['video_tasks']]):
                    _, _, frames = collect(vector, model, [target],
                        1_600_000_000 + index * 100003, record=True)
                    for t in range(cfg['horizon']):
                        canvas = Image.new('RGB', (320 * K, 354), '#12141c')
                        draw = ImageDraw.Draw(canvas)
                        draw.text((12, 4), f'RL2 + PPO: {OBJETIVOS[target]}', fill='white')
                        for ep in range(K):
                            canvas.paste(Image.fromarray(frames[ep][t]).resize((320, 320)), (320 * ep, 34))
                            draw.text((320 * ep + 12, 19), f'episodio {ep + 1}', fill='white')
                        movie.append_data(np.asarray(canvas))
        finally:
            vector.close()
    return summary


def parse_config():
    parser = argparse.ArgumentParser(description='Crafter: RL² + PPO recorrente; código independente.')
    parser.add_argument('--smoke', action='store_true', help='2 atualizações curtas; não mede aprendizado.')
    parser.add_argument('--iterations', type=int, default=320)
    parser.add_argument('--ppo-epochs', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=192)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--envs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-minutes', type=float, default=120)
    parser.add_argument('--output', default='resultados_crafter_rl2_ppo', help='Pasta raiz; cada execução cria subpasta.')
    parser.add_argument('--resume', default='', help='last.pt desta versão; preservar parâmetros de treino.')
    parser.add_argument('--checkpoint', default='', help='best.pt ou last.pt desta versão para --eval-only.')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--video-tasks', type=int, default=4)
    parser.add_argument('--test-repeats', type=int, default=8, help='Repetições por cada um dos 4 objetivos.')
    parser.add_argument('--validate-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--target-kl', type=float, default=0.02)
    parser.add_argument('--reward-cap', type=int, default=None,
                        help='Máximo de eventos do objetivo recompensados por episódio (0 = sem teto, como na v1; 1 = só o 1º).')
    parser.add_argument('--wrong-penalty', type=float, default=0.0,
                        help='Penalidade a cada evento de um dos outros 3 objetivos (padrão 0).')
    parser.add_argument('--reward', choices=['eventos', 'primeiro'], default=None,
                        help='Atalho: eventos = --reward-cap 0; primeiro = --reward-cap 1.')
    parser.add_argument('--episodes', type=int, default=3, help='Episódios por trial (K).')
    parser.add_argument('--no-memory', action='store_true',
                        help='Controle sem memória: a GRU recebe estado zerado a cada passo.')
    cfg = vars(parser.parse_args())
    atalho = cfg.pop('reward')
    if atalho is not None:
        teto = {'eventos': 0, 'primeiro': 1}[atalho]
        if cfg['reward_cap'] is not None and cfg['reward_cap'] != teto:
            parser.error('--reward e --reward-cap são contraditórios.')
        cfg['reward_cap'] = teto
    if cfg['reward_cap'] is None:
        cfg['reward_cap'] = 0
    if cfg['reward_cap'] < 0 or not math.isfinite(cfg['wrong_penalty']) or cfg['wrong_penalty'] < 0:
        parser.error('Exige --reward-cap >= 0 e --wrong-penalty >= 0.')
    if not 1 <= cfg['episodes'] <= 10:
        parser.error('--episodes deve estar entre 1 e 10.')
    if cfg['smoke']:
        cfg.update(iterations=2, ppo_epochs=2, horizon=32, hidden=64, test_repeats=1,
                   validate_every=1, video_tasks=1)
    positive = ('iterations', 'ppo_epochs', 'horizon', 'hidden', 'envs', 'test_repeats',
                'validate_every', 'max_minutes', 'video_tasks', 'lr', 'max_grad_norm', 'target_kl')
    if any(not math.isfinite(cfg[key]) or cfg[key] <= 0 for key in positive):
        parser.error('Orçamentos e taxas devem ser positivos e finitos.')
    # Bound seed offsets so train, validation and test ranges cannot overlap.
    if cfg['envs'] > 1024 or cfg['test_repeats'] > 250 or cfg['horizon'] > 100_000:
        parser.error('Limites suportados: envs <= 1024, test-repeats <= 250, horizon <= 100000.')
    if not 0 <= cfg['seed'] < 2**32:
        parser.error('seed deve estar em [0, 2**32).')
    if not 0 < cfg['gamma'] <= 1 or not 0 <= cfg['gae_lambda'] <= 1 or not 0 < cfg['clip'] < 1:
        parser.error('Exige 0 < gamma <= 1, 0 <= gae-lambda <= 1 e 0 < clip < 1.')
    if any(not math.isfinite(cfg[key]) or cfg[key] < 0 for key in ('entropy_coef', 'value_coef')):
        parser.error('Coeficientes devem ser finitos e não negativos.')
    if cfg['eval_only'] and not cfg['checkpoint']:
        parser.error('--eval-only exige --checkpoint.')
    if cfg['resume'] and (cfg['eval_only'] or cfg['checkpoint']):
        parser.error('--resume é exclusivo para retomar treino.')
    if cfg['checkpoint'] and not cfg['eval_only']:
        parser.error('--checkpoint exige --eval-only; para retomar treino use --resume.')
    if cfg['device'] == 'auto':
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA indisponível; selecione GPU ou use --device cpu.')
    return cfg


def main():
    cfg = parse_config()
    global TETO_RECOMPENSA, PENALIDADE_ERRADO, K
    TETO_RECOMPENSA, PENALIDADE_ERRADO, K = cfg['reward_cap'], cfg['wrong_penalty'], cfg['episodes']
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    output = Path(cfg['output']).expanduser().resolve() / (time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    manifest = dict(version=VERSION, algorithm='RL² + recurrent PPO', python=platform.python_version(),
        platform=platform.platform(), device=cfg['device'], episodes_per_trial=K,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        packages={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'crafter')},
        observation='local semantic 9x7 + inventory/direction/daylight + public clocks',
        reward=(('+1 per hidden-goal event' if cfg['reward_cap'] == 0
                 else f"+1 per hidden-goal event, at most {cfg['reward_cap']} per episode")
                + (f"; -{cfg['wrong_penalty']} per event of another goal" if cfg['wrong_penalty'] else '')
                + '; no shaping or goal label'),
        memory='none (control: GRU state reset every step)' if cfg['no_memory']
               else 'GRU memory kept across the episodes of a trial',
        evaluation='New world seeds; same four goal categories as training',
        claim='Smoke verifies execution, not learning or superiority; no teacher or imitation.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Resultados:', output, flush=True)
    try:
        if cfg['eval_only']:
            state = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
            check_checkpoint(state, cfg)
            model = Agente(cfg['hidden'], sem_memoria=cfg['no_memory'])
            model.load_state_dict(state['model'])
        else:
            model = train(cfg, output)
        summary = evaluate(cfg, model, output)
        status = dict(status='completed', completed_updates=cfg.get('completed_updates'),
                      training_complete=cfg['eval_only'] or cfg.get('completed_updates', 0) == cfg['iterations'])
        (output / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        print('Método          ' + ''.join(f'   ep.{k}' for k in range(1, K + 1)) + '  (sucesso real)')
        for mode in ('memory', 'reset', 'untrained', 'random'):
            print(f'{mode:16s}', ' '.join(f'{r["success"]:6.1%}' for r in summary if r['mode'] == mode))
        print('Smoke não mede convergência. Um treino/seed não demonstra superioridade geral.')
    except BaseException as error:
        (output / 'status.json').write_text(json.dumps(dict(status='failed', error=repr(error))), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
