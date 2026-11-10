"""Analisa várias execuções de crafter_meta_goals_rl2_ppo (v1, v2 ou v3) de uma vez.

Uso:
    python analisar.py PASTA [PASTA ...] [--saida analise] [--referencia A-R2]

Cada PASTA pode ser uma execução (contém episodes.csv) ou uma pasta acima delas.
Agrupamento: se a execução tem um experimento.json (gravado por experimentos.py),
o grupo é o id do experimento (ex.: A-R2); senão, os hiperparâmetros que variam.
Execuções com a mesma configuração e a mesma semente são a mesma execução
retomada: só a última completa conta.

Gera, na pasta de saída:
    sucesso_por_modo.csv        sucesso e passos até o sucesso por grupo × modo × episódio
    sucesso_por_objetivo.csv    o mesmo, separado pelos 4 objetivos
    efeito_memoria.csv          memory − reset por episódio, pareado por tarefa, IC 95%
    comparacao_com_referencia.csv  (com --referencia) diferenças de cada grupo para a referência
    comportamento_por_grupo.csv seletividade e bebidas fora da água (se houver comportamento_resumo.json)
    validacao.csv               sucesso médio nas 5 últimas validações (para escolher configurações)
    curvas_treino.png           sucesso de treino, entropia e validação por grupo
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HIPERPARAMETROS = ('reward_cap', 'wrong_penalty', 'episodes', 'no_memory', 'envs', 'iterations',
                   'horizon', 'hidden', 'ppo_epochs', 'lr', 'gamma', 'gae_lambda', 'clip',
                   'entropy_coef', 'value_coef', 'max_grad_norm', 'target_kl')
MODOS = ('memory', 'reset', 'untrained', 'random')
OBJETIVOS = ('collect_wood', 'collect_drink', 'eat_cow', 'collect_sapling')


def normalizar(cfg):
    """Configurações da v1/v2 não têm os campos da v3: preenche com o comportamento delas."""
    cfg = dict(cfg)
    if 'reward_cap' not in cfg:
        cfg['reward_cap'] = 1 if cfg.get('reward') == 'primeiro' else 0
    cfg.setdefault('wrong_penalty', 0.0)
    cfg.setdefault('episodes', 3)
    cfg.setdefault('no_memory', False)
    return cfg


def ler_json(caminho, padrao=None):
    try:
        return json.loads(Path(caminho).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return padrao


def carregar(execucao):
    cfg = normalizar(ler_json(execucao / 'config.json', {}))
    with (execucao / 'episodes.csv').open(encoding='utf-8') as f:
        linhas = list(csv.DictReader(f))
    for r in linhas:
        r['episode'], r['task_id'] = int(r['episode']), int(r['task_id'])
        r['success'], r['return_'] = int(r['success']), float(r['return_'])
        r['first_success'] = int(r['first_success'])
    rotulo = ler_json(execucao / 'experimento.json') or ler_json(execucao.parent / 'experimento.json')
    status = ler_json(execucao / 'status.json', {})
    return dict(pasta=execucao, cfg=cfg, linhas=linhas, historico=ler_json(execucao / 'training.json', []),
                rotulo=(rotulo or {}).get('id'), completo=bool(status.get('training_complete')),
                comportamento=ler_json(execucao / 'comportamento_resumo.json'))


def encontrar_execucoes(pastas):
    achadas = set()
    for pasta in pastas:
        for csv_path in Path(pasta).rglob('episodes.csv'):
            if (csv_path.parent / 'config.json').exists():
                achadas.add(csv_path.parent.resolve())
    return sorted(achadas)


def deduplicar(execucoes):
    """Mesma configuração + mesma semente = mesma execução (retomada). Fica a última completa."""
    por_chave = defaultdict(list)
    for e in execucoes:
        chave = (e['rotulo'], tuple(str(e['cfg'].get(k)) for k in HIPERPARAMETROS), e['cfg'].get('seed'))
        por_chave[chave].append(e)
    escolhidas = []
    for chave, lista in por_chave.items():
        lista.sort(key=lambda e: (e['completo'], e['pasta'].name))
        escolhida = lista[-1]
        if not escolhida['completo']:
            print(f'Aviso: treino incompleto em {escolhida["pasta"]} (avaliação de um modelo parcial).')
        escolhidas.append(escolhida)
    return escolhidas


def nomear_grupos(execucoes):
    sem_rotulo = [e for e in execucoes if not e['rotulo']]
    chaves = {id(e): tuple(e['cfg'].get(k) for k in HIPERPARAMETROS) for e in sem_rotulo}
    unicos = sorted(set(chaves.values()), key=str)
    variam = [i for i, k in enumerate(HIPERPARAMETROS) if len({u[i] for u in unicos}) > 1]
    for e in execucoes:
        if e['rotulo']:
            e['grupo'] = e['rotulo']
        else:
            u = chaves[id(e)]
            e['grupo'] = ' '.join(f'{HIPERPARAMETROS[i]}={u[i]}' for i in variam) or 'padrao'


def ic_bootstrap(valores, n=5000, semente=0):
    valores = np.asarray(valores, dtype=float)
    if len(valores) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(semente)
    medias = rng.choice(valores, size=(n, len(valores)), replace=True).mean(axis=1)
    return float(valores.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def ic_diferenca(a, b, n=5000, semente=0):
    """IC 95% de mean(a) − mean(b), reamostrando os dois grupos de forma independente."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(semente)
    da = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    db = rng.choice(b, size=(n, len(b)), replace=True).mean(axis=1)
    d = da - db
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def escrever_csv(caminho, linhas):
    if not linhas:
        return
    campos = list(dict.fromkeys(k for linha in linhas for k in linha))
    with caminho.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)


def valores_por_tarefa(execucoes, modo, ep, metrica):
    """Lista de valores (um por tarefa e semente) de um modo e episódio; passos em fração do horizonte."""
    saida = []
    for e in execucoes:
        h = e['cfg'].get('horizon', 192)
        for r in e['linhas']:
            if r['mode'] == modo and r['episode'] == ep:
                saida.append(r['success'] if metrica == 'success' else r['first_success'] / h)
    return saida


def diferencas_pareadas(execucoes, ep, metrica):
    difs = []
    for e in execucoes:
        m = {r['task_id']: r[metrica] for r in e['linhas'] if r['mode'] == 'memory' and r['episode'] == ep}
        s = {r['task_id']: r[metrica] for r in e['linhas'] if r['mode'] == 'reset' and r['episode'] == ep}
        difs += [m[t] - s[t] for t in m if t in s]
    return difs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pastas', nargs='+')
    parser.add_argument('--saida', default='analise')
    parser.add_argument('--referencia', default='', help='grupo de referência para as comparações (ex.: A-R2)')
    args = parser.parse_args()
    execucoes = [carregar(p) for p in encontrar_execucoes(args.pastas)]
    if not execucoes:
        raise SystemExit('Nenhuma execução com episodes.csv e config.json encontrada.')
    execucoes = deduplicar(execucoes)
    nomear_grupos(execucoes)
    grupos = defaultdict(list)
    for e in execucoes:
        grupos[e['grupo']].append(e)
    nomes = sorted(grupos)
    episodios = {g: sorted({r['episode'] for e in grupos[g] for r in e['linhas']}) for g in nomes}
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    print(f'{len(execucoes)} execuções em {len(nomes)} grupo(s): {", ".join(nomes)}\n')

    # 1) Sucesso e passos por modo × episódio: média por semente, depois média e desvio entre sementes.
    tabela_modo, tabela_obj = [], []
    for g in nomes:
        for modo in MODOS:
            for ep in episodios[g]:
                for obj in (None,) + OBJETIVOS:
                    taxas, passos = [], []
                    for e in grupos[g]:
                        sel = [r for r in e['linhas'] if r['mode'] == modo and r['episode'] == ep
                               and (obj is None or r['target'] == obj)]
                        if sel:
                            taxas.append(np.mean([r['success'] for r in sel]))
                            passos.append(np.mean([r['first_success'] for r in sel]))
                    if not taxas:
                        continue
                    linha = dict(grupo=g, modo=modo, episodio=ep, sementes=len(taxas),
                                 sucesso_medio=round(float(np.mean(taxas)), 4),
                                 desvio_entre_sementes=round(float(np.std(taxas)), 4),
                                 passos_ate_sucesso=round(float(np.mean(passos)), 1))
                    (tabela_modo if obj is None else tabela_obj).append(linha if obj is None else dict(linha, objetivo=obj))
    escrever_csv(saida / 'sucesso_por_modo.csv', tabela_modo)
    escrever_csv(saida / 'sucesso_por_objetivo.csv', tabela_obj)
    for g in nomes:
        print(f'== {g} ({len(grupos[g])} semente(s)) ==')
        print(f'{"modo":10s}' + ''.join(f'      ep.{ep}       ' for ep in episodios[g]))
        for modo in MODOS:
            cel = {r['episodio']: r for r in tabela_modo if r['grupo'] == g and r['modo'] == modo}
            if cel:
                print(f'{modo:10s}' + ''.join(f'  {cel[ep]["sucesso_medio"]:6.1%} ±{cel[ep]["desvio_entre_sementes"]:5.1%}    '
                                               for ep in episodios[g] if ep in cel))
        print('memory por objetivo:', ' | '.join(
            obj.split('_')[1] + ' ' + '→'.join(f'{r["sucesso_medio"]:.0%}' for r in tabela_obj
                                                if r['grupo'] == g and r['modo'] == 'memory' and r['objetivo'] == obj)
            for obj in OBJETIVOS))
        print()

    # 2) Efeito da memória: memory − reset, pareado por (execução, tarefa).
    efeito = []
    for g in nomes:
        for ep in episodios[g]:
            for metrica in ('success', 'first_success'):
                media, baixo, alto = ic_bootstrap(diferencas_pareadas(grupos[g], ep, metrica))
                efeito.append(dict(grupo=g, episodio=ep, metrica=metrica,
                                   pares=len(diferencas_pareadas(grupos[g], ep, metrica)),
                                   memory_menos_reset=round(media, 4), ic95_baixo=round(baixo, 4),
                                   ic95_alto=round(alto, 4)))
    escrever_csv(saida / 'efeito_memoria.csv', efeito)
    print('Efeito da memória (memory − reset), pareado por tarefa, IC 95%  [no ep.1 é sempre 0]')
    for g in nomes:
        for ep in episodios[g][1:]:
            s = next(r for r in efeito if r['grupo'] == g and r['episodio'] == ep and r['metrica'] == 'success')
            p = next(r for r in efeito if r['grupo'] == g and r['episodio'] == ep and r['metrica'] == 'first_success')
            print(f'  {g:14s} ep.{ep}: sucesso {s["memory_menos_reset"]:+6.1%} [{s["ic95_baixo"]:+6.1%}, {s["ic95_alto"]:+6.1%}]'
                  f' | passos {p["memory_menos_reset"]:+6.1f} [{p["ic95_baixo"]:+6.1f}, {p["ic95_alto"]:+6.1f}]')
    print('  (passos negativos = memory acerta antes)\n')

    # 3) Comparação com a referência.
    if args.referencia:
        if args.referencia not in grupos:
            raise SystemExit(f'Referência {args.referencia!r} não encontrada. Grupos: {", ".join(nomes)}')
        ref = grupos[args.referencia]
        comparacao = []
        for g in nomes:
            if g == args.referencia:
                continue
            for ep in sorted(set(episodios[g]) & set(episodios[args.referencia])):
                linha = dict(grupo=g, referencia=args.referencia, episodio=ep)
                for nome, a, b in (
                        ('sucesso_memory', valores_por_tarefa(grupos[g], 'memory', ep, 'success'),
                         valores_por_tarefa(ref, 'memory', ep, 'success')),
                        ('fracao_do_episodio_ate_sucesso', valores_por_tarefa(grupos[g], 'memory', ep, 'first_success'),
                         valores_por_tarefa(ref, 'memory', ep, 'first_success')),
                        ('efeito_memoria_sucesso', diferencas_pareadas(grupos[g], ep, 'success'),
                         diferencas_pareadas(ref, ep, 'success'))):
                    d, lo, hi = ic_diferenca(a, b)
                    linha.update({f'delta_{nome}': round(d, 4), f'{nome}_ic95_baixo': round(lo, 4),
                                  f'{nome}_ic95_alto': round(hi, 4)})
                comparacao.append(linha)
        escrever_csv(saida / 'comparacao_com_referencia.csv', comparacao)
        print(f'Diferença para a referência {args.referencia} (grupo − referência, IC 95%):')
        def fmt(r, nome):
            return (f'{r["delta_" + nome]:+6.1%} [{r[nome + "_ic95_baixo"]:+6.1%}, '
                    f'{r[nome + "_ic95_alto"]:+6.1%}]')
        for r in comparacao:
            print(f'  {r["grupo"]:10s} ep.{r["episodio"]}: sucesso de memory {fmt(r, "sucesso_memory")} | '
                  f'efeito da memória {fmt(r, "efeito_memoria_sucesso")} | '
                  f'fração do episódio até o sucesso {fmt(r, "fracao_do_episodio_ate_sucesso")}')
        print()

    # 4) Comportamento (seletividade), se o diagnóstico rodou.
    comportamento = []
    for g in nomes:
        resumos = [e['comportamento'] for e in grupos[g] if e['comportamento']]
        if not resumos:
            continue
        for modo in ('memory', 'reset'):
            for ep in episodios[g]:
                vals = [r[modo][str(ep)] for r in resumos if str(ep) in r.get(modo, {})]
                if vals:
                    comportamento.append(dict(
                        grupo=g, modo=modo, episodio=ep, sementes=len(vals),
                        seletividade=round(float(np.mean([v['seletividade'] for v in vals])), 4),
                        seletividade_desvio=round(float(np.std([v['seletividade'] for v in vals])), 4),
                        acerto=round(float(np.mean([v['acerto'] for v in vals])), 4),
                        erro=round(float(np.mean([v['erro'] for v in vals])), 4),
                        bebidas_fora_agua=round(float(np.mean([v['bebidas_fora_agua'] for v in vals])), 2)))
    escrever_csv(saida / 'comportamento_por_grupo.csv', comportamento)
    if comportamento:
        print('Comportamento (modo memory): seletividade por episódio | bebidas fora da água no último episódio')
        for g in nomes:
            sel = [r for r in comportamento if r['grupo'] == g and r['modo'] == 'memory']
            if sel:
                print(f'  {g:14s} ' + ' → '.join(f'{r["seletividade"]:+.2f}' for r in sel)
                      + f' | {sel[-1]["bebidas_fora_agua"]:.1f}')
        print()

    # 5) Validação: média das 5 últimas validações (para escolher configurações sem olhar o teste).
    validacao = []
    for g in nomes:
        por_semente = []
        for e in grupos[g]:
            v = [r['validation_success'] for r in e['historico'] if 'validation_success' in r]
            if v:
                por_semente.append(float(np.mean(v[-5:])))
        if por_semente:
            validacao.append(dict(grupo=g, sementes=len(por_semente),
                                  validacao_media=round(float(np.mean(por_semente)), 4),
                                  validacao_desvio=round(float(np.std(por_semente)), 4)))
    validacao.sort(key=lambda r: -r['validacao_media'])
    escrever_csv(saida / 'validacao.csv', validacao)
    print('Ranking pela validação (sucesso no último episódio, média das 5 últimas validações):')
    for i, r in enumerate(validacao, 1):
        print(f'  {i:2d}. {r["grupo"]:14s} {r["validacao_media"]:6.1%} ± {r["validacao_desvio"]:.1%}')
    print()

    # 6) Curvas de treino por grupo (média entre sementes).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figura, eixos = plt.subplots(1, 3, figsize=(16, 4.2))
    for g in nomes:
        historicos = [e['historico'] for e in grupos[g] if e['historico']]
        if not historicos:
            continue
        n = min(len(h) for h in historicos)
        janela = max(1, n // 30)
        for eixo, campo in zip(eixos[:2], ('train_success', 'entropy')):
            serie = np.mean([[r[campo] for r in h[:n]] for h in historicos], axis=0)
            suave = np.convolve(serie, np.ones(janela) / janela, mode='valid')
            eixo.plot(np.arange(len(suave)) + janela, suave, label=g)
        pontos = defaultdict(list)
        for h in historicos:
            for r in h:
                if 'validation_success' in r:
                    pontos[r['iteration']].append(r['validation_success'])
        its = sorted(pontos)
        eixos[2].plot(its, [np.mean(pontos[i]) for i in its], marker='.', label=g)
    for eixo, titulo in zip(eixos, ('Sucesso nos trials de treino (média móvel)', 'Entropia da política',
                                    'Sucesso na validação (último episódio, 4 tarefas)')):
        eixo.set_title(titulo)
        eixo.set_xlabel('Iteração')
        eixo.grid(alpha=0.3)
    eixos[0].legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(saida / 'curvas_treino.png', dpi=130)
    plt.close(figura)
    print(f'Arquivos em {saida.resolve()}')


if __name__ == '__main__':
    main()
