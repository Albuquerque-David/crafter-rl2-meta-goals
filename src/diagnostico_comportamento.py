"""Matriz de comportamento: o que o agente faz em cada objetivo, episódio a episódio.

Para cada objetivo sorteado (linha) e episódio, conta quantas vezes o agente cumpriu
CADA um dos 4 objetivos (colunas). Um agente que identifica o objetivo concentra os
eventos no objetivo certo, cada vez mais do primeiro para o último episódio. Um agente
com estratégia fixa faz a mesma coisa em todas as linhas.

Resumo por episódio (modos memory e reset):
    acerto        fração de tarefas em que cumpriu o objetivo sorteado
    erro          fração média dos outros 3 objetivos que também cumpriu
    seletividade  acerto - erro  (~0 para "fazer tudo" ou para o acaso; > 0 quando
                  o agente escolhe o objetivo certo)
    bebidas_fora_agua  vezes que bebeu água por episódio em tarefas que não são de água

Uso (com o mesmo Python do treino):
    python diagnostico_comportamento.py --script crafter_meta_goals_rl2_ppo_v3.py \
        --checkpoint resultados/.../best.pt [--repeats 8] [--envs 8]

Grava comportamento.csv e comportamento_resumo.json na pasta do checkpoint (ou em
--saida). Usa as mesmas sementes de mundo e de sorteio da avaliação do script, então
os números do modo memory batem com o episodes.csv. Funciona com checkpoints da v1,
v2 e v3: recompensa, número de episódios e memória vêm do próprio checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def carregar_modulo(caminho):
    spec = importlib.util.spec_from_file_location('rl2_script', caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def configurar(m, contrato):
    """Reproduz no módulo a recompensa e o número de episódios do treino."""
    if hasattr(m, 'TETO_RECOMPENSA'):                       # v3
        m.TETO_RECOMPENSA = contrato.get('reward_cap', 0)
        m.PENALIDADE_ERRADO = contrato.get('wrong_penalty', 0.0)
    if hasattr(m, 'MODO_RECOMPENSA'):                       # v2
        m.MODO_RECOMPENSA = contrato.get('reward', 'eventos')
    m.K = contrato.get('episodes_per_trial', 3)
    hidden = contrato['hidden']
    if contrato.get('no_memory', False):
        return m.Agente(hidden, sem_memoria=True)
    return m.Agente(hidden)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--script', required=True, help='o script usado no treino (v1, v2 ou v3)')
    parser.add_argument('--checkpoint', required=True, help='best.pt ou last.pt')
    parser.add_argument('--repeats', type=int, default=8, help='tarefas por objetivo')
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--saida', default='', help='pasta de saída (padrão: a pasta do checkpoint)')
    args = parser.parse_args()

    m = carregar_modulo(args.script)
    estado = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if estado.get('version') != m.VERSION:
        raise SystemExit(f"Checkpoint da versão {estado.get('version')}, script da versão {m.VERSION}.")
    contrato = estado['contract']
    modelo = configurar(m, contrato)
    modelo.load_state_dict(estado['model'])
    modelo.eval()
    horizonte, k = contrato['horizon'], m.K

    # Guarda as conquistas de cada episódio antes de o ambiente gerar o mundo seguinte.
    reset_original = m.TaskEnv.reset

    def reset_registrando(self, target, seed):
        if self.game.env is not None:
            self.registro.append(dict(self.game.env._player.achievements))
        return reset_original(self, target, seed)

    m.TaskEnv.reset = reset_registrando

    objetivos = list(m.OBJETIVOS)
    alvos = list(range(len(objetivos))) * args.repeats
    contagem = defaultdict(list)          # (modo, objetivo sorteado, episódio, conquista) -> [eventos]
    for modo in ('memory', 'reset'):
        vetor = m.Vetor(args.envs, horizonte)
        torch.manual_seed(54321)          # mesmo sorteio de ações da avaliação do script
        for lo in range(0, len(alvos), vetor.n):
            lote = alvos[lo:lo + vetor.n]
            reais = len(lote)
            lote += [lote[0]] * (vetor.n - reais)
            for env in vetor.envs:
                env.game.env = None
                env.registro = []
            m.collect(vetor, modelo, lote, 1_600_000_000 + lo * 100003, reset_memory=modo == 'reset')
            for b, env in enumerate(vetor.envs[:reais]):
                episodios = env.registro + [dict(env.game.env._player.achievements)]
                for ep, conquistas in enumerate(episodios, start=1):
                    for nome in objetivos:
                        contagem[(modo, objetivos[lote[b]], ep, nome)].append(conquistas.get(nome, 0))
        vetor.close()

    linhas = []
    for (modo, alvo, ep, nome), v in sorted(contagem.items()):
        linhas.append(dict(modo=modo, objetivo_sorteado=alvo, episodio=ep, conquista=nome,
                           eventos_medios=round(float(np.mean(v)), 3),
                           fracao_com_evento=round(float(np.mean(np.asarray(v) > 0)), 3), n=len(v)))

    # Resumo por episódio: acerto, erro, seletividade e bebidas fora de tarefas de água.
    resumo = {}
    for modo in ('memory', 'reset'):
        resumo[modo] = {}
        for ep in range(1, k + 1):
            acertos, erros, bebidas = [], [], []
            for alvo in objetivos:
                certo = np.asarray(contagem[(modo, alvo, ep, alvo)]) > 0
                errados = np.mean([np.asarray(contagem[(modo, alvo, ep, o)]) > 0
                                   for o in objetivos if o != alvo], axis=0)
                acertos += certo.tolist()
                erros += errados.tolist()
                if alvo != 'collect_drink':
                    bebidas += contagem[(modo, alvo, ep, 'collect_drink')]
            resumo[modo][str(ep)] = dict(
                acerto=round(float(np.mean(acertos)), 4), erro=round(float(np.mean(erros)), 4),
                seletividade=round(float(np.mean(acertos) - np.mean(erros)), 4),
                bebidas_fora_agua=round(float(np.mean(bebidas)), 3), tarefas=len(acertos))

    saida = Path(args.saida) if args.saida else Path(args.checkpoint).resolve().parent
    saida.mkdir(parents=True, exist_ok=True)
    with (saida / 'comportamento.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    (saida / 'comportamento_resumo.json').write_text(
        json.dumps(dict(checkpoint=str(Path(args.checkpoint).resolve()), repeats=args.repeats, **resumo),
                   indent=2), encoding='utf-8')

    curto = {'collect_wood': 'madeira', 'collect_drink': 'agua', 'eat_cow': 'vaca', 'collect_sapling': 'muda'}
    for modo in ('memory', 'reset'):
        print(f'\n=== modo {modo}: eventos médios por episódio (linhas = objetivo sorteado) ===')
        print(f'{"objetivo":10s} {"ep":>2s}' + ''.join(f'{curto[n]:>9s}' for n in objetivos))
        for alvo in objetivos:
            for ep in range(1, k + 1):
                valores = [np.mean(contagem[(modo, alvo, ep, n)]) for n in objetivos]
                print(f'{curto[alvo]:10s} {ep:2d}' + ''.join(f'{v:9.2f}' for v in valores))
    print('\nResumo por episódio   acerto   erro   seletividade   bebidas fora da água')
    for modo in ('memory', 'reset'):
        for ep in range(1, k + 1):
            r = resumo[modo][str(ep)]
            print(f'  {modo:7s} ep.{ep}         {r["acerto"]:5.1%}  {r["erro"]:5.1%}      {r["seletividade"]:+.2f}'
                  f'          {r["bebidas_fora_agua"]:6.2f}')
    print(f'\nArquivos em {saida}')


if __name__ == '__main__':
    main()
