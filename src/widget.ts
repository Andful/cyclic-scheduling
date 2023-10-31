// Copyright (c) Andrea Nardi
// Distributed under the terms of the Modified BSD License.

import {
  DOMWidgetModel,
  DOMWidgetView,
  ISerializers,
  WidgetModel,
  WidgetView,
} from '@jupyter-widgets/base';

import * as d3 from "d3";

import { MODULE_NAME, MODULE_VERSION } from './version';

import '../css/widget.css';

export class CyclicSchedulingPlotModel extends DOMWidgetModel {
  defaults() {
    return {
      ...super.defaults(),
      _model_name: CyclicSchedulingPlotModel.model_name,
      _model_module: CyclicSchedulingPlotModel.model_module,
      _model_module_version: CyclicSchedulingPlotModel.model_module_version,
      _view_name: CyclicSchedulingPlotModel.view_name,
      _view_module: CyclicSchedulingPlotModel.view_module,
      _view_module_version: CyclicSchedulingPlotModel.view_module_version,
      data: '',
    };
  }

  static serializers: ISerializers = {
    ...DOMWidgetModel.serializers,
    // Add any extra serializers here
  };

  static model_name = 'CyclicSchedulingPlotModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'CyclicSchedulingPlotView'; // Set to null if no view
  static view_module = MODULE_NAME; // Set to null if no view
  static view_module_version = MODULE_VERSION;
}

interface Actor {
  name: string,
  math_name: string,
  execution_time: number,
  processor: string,
  color: string,
}

interface Data {
  actor2processors: Array<number>,
  additional_channels: Array<unknown>,
  cycle_time: number,
  problem: {
    actors: Array<Actor>,
    channels: Array<{
      source: number,
      targer: number,
      initial_tokens: number,
    }>,
    processors: { [index: string]: number }
  },
  processors: Array<string>,
  t: Array<number>
}

export class CyclicSchedulingPlotView extends DOMWidgetView {
  initialize(parameters: WidgetView.IInitializeParameters<WidgetModel>): void {
    const zip = <K, E>(a: K[], b: E[]): [K, E][] => a.map((k, i) => [k, b[i]]);
    const cartesianProduct = <K, E>(a: K[], b: E[]): [K, E][] => a.map((k) => b.map((e) => [k, e])).flat(1) as [K, E][];
    const range = (s: number, e: number) => [...Array(e - s)].map((_, i) => i + s)

    const marginTop = 20;
    const marginBottom = 30;
    const marginLeft = 80;

    const data = JSON.parse(this.model.get("data")) as Data;
    const minT = Math.min(...data.t);
    data.t = data.t.map((e) => e - minT);
    const maxT = Math.max(...data.t.map((e, i) => e + data.problem.actors[i].execution_time));
    this.el.classList.add("plot-container");
    const svg = d3.select(this.el).append("svg");
  
    const x = d3.scaleLinear()
    const y = d3.scaleBand<number>()
      .domain(d3.range(data.processors.length))
      .padding(0.1);

    const xAxis = svg.append("g");
    const bars = svg.append("g");
    const blankRectangle = svg.append("rect")
      .attr("x", 0)
      .attr("y", marginTop)
      .attr("width", marginLeft)
      .attr("fill", "white");
    const yAxis = svg.append("g");
    const dragPanel = svg
      .append("rect")
      .attr("class", "drag-panel")
      .attr("opacity", 0)
      .attr("x", marginLeft)
      .attr("y", marginTop)

    let offset = 0;
    let windowWidth = 3 * data.cycle_time;
    const getLower = () => Math.ceil((- maxT + offset)/data.cycle_time);
    const getUpper = () => Math.ceil((windowWidth + offset)/data.cycle_time)
    const plot = () => {
      const lower = getLower();
      const upper = getUpper();

      const width = this.el.clientWidth
      const height = this.el.clientHeight
      
      x.domain([offset, offset + windowWidth]).range([marginLeft, width]);
      y.rangeRound([marginTop, height - marginBottom])

      svg.attr("width", width)
        .attr("height", height);
      
      dragPanel
        .attr("width", width - marginLeft)
        .attr("height", height - marginTop - marginBottom);

      xAxis
        .attr("transform", `translate(0,${height - marginBottom})`)
        .html(null)
        .call(d3.axisBottom(x));

      const activations = cartesianProduct(zip(data.problem.actors, data.t).map(([a,b], c) => [a, b, c] as [Actor, number, number]), range(lower, upper)).map(([[a, b, c], d]) => [a, b, c, d] as [Actor, number, number, number])
      const cycle_time = data.cycle_time;
      bars.selectAll(".bar")
        .data(activations)
        .enter()
        .append("g")
        .attr("class", "bar")
        .each(function ([a, t, k]) {
          const elem = d3
          .select(this)
          .append("g")

          elem
          .append("rect")
          .attr("x", 0)
          .attr("y", 0)
          .attr("fill", a.color)
          .attr("stroke", "black")
          .attr("stroke-width", "1")

          elem
          .append("text")
          .attr("dominant-baseline", "text-before-edge")
          .attr("x", 0)
          .attr("y", 0);
        });

      const barContent = bars.selectAll(".bar")
        .data(activations);

      barContent
        .exit()
        .remove()

      barContent
        .select("g")
        .attr("transform",([_,t,i,k]) => `translate(${x(t + k * cycle_time)},${y(data.actor2processors[i]) ?? 0})`)
        .each(function ([a,t, _,k]) {
          const elem = d3
            .select(this);

          elem
            .select("rect")
            .attr("fill", a.color)
            .attr("width", x(t + a.execution_time) - x(t))
            .attr("height", y.bandwidth())
            .attr("opacity", k >= 0 ? 1 : 0.5);
          
          elem
          .select("text")
          .text(`${a.name}(${k})`)
        })

      blankRectangle
        .attr("height", height - marginTop - marginBottom);

      yAxis
        .attr("transform", `translate(${marginLeft},0)`)
        .html(null)
        .call(d3.axisLeft(y).tickFormat((d) => data.processors[d]));
    };

    dragPanel
      .on("wheel", (event: WheelEvent) => {
        event.preventDefault();
        windowWidth = Math.max(Math.min(windowWidth + data.cycle_time * event.deltaY/1000, 10 * data.cycle_time), data.cycle_time/10);
        plot();
      })
      .on("mousedown", (startEvent: MouseEvent) => {
      const startOffset = offset;
      const elementWidth = this.el.clientWidth;
      const mousemove = (event: MouseEvent) => {
        offset = startOffset + (startEvent.clientX - event.clientX) * windowWidth / (elementWidth - marginLeft);
        plot()
      }
      const mouseup = () => {
        d3.select(document.body)
        .attr("data-dragging", null)
        window.removeEventListener("mouseup", mouseup);
        window.removeEventListener("mousemove", mousemove);
      }
      d3.select(document.body)
        .attr("data-dragging", 1)
        window.addEventListener("mouseup", mouseup);
        window.addEventListener("mousemove", mousemove);
    })

    const mo = new MutationObserver(() => {
      if (!document.contains(this.el)) {
        return;
      }
      plot();
      mo.disconnect();
    });
    mo.observe(document, { attributes: false, childList: true, characterData: false, subtree: true });
    window.addEventListener('resize', plot);
    this.toRemove = plot
  }

  toRemove: (() => void) | null = null

  remove(): void {
    if (this.toRemove != null) {
      window.removeEventListener('resize', this.toRemove)
    }
  }
}
