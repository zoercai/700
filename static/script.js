/**
 * Created by Zoe on 13/08/16.
 */

function loadData() {
    // get values
    var startDate = $('#startDate').val();
    var endDate = $('#endDate').val();

    visualise(startDate, endDate);

    return false;
};

function visualise(startDate, endDate) {
    var json_source = $SCRIPT_ROOT + '/cluster?start_date=' + startDate + '&end_date=' + endDate
    // var json_source = $SCRIPT_ROOT + '/static/miserables.json'
    // json_source = $SCRIPT_ROOT + '/static/output.json'

    d3.json(json_source, function (error, graph) {
        var width = +window.innerWidth;
        var height = +window.innerHeight;
        var zoom = d3.zoom().scaleExtent([.2, 10]).on("zoom", zoomed);

        // var svg = d3.select("svg"),
        //     width = +svg.attr("width"),
        //     height = +svg.attr("height");

        var svg = d3.select("svg").attr("viewBox", "0 0 " + width + " " + height ).attr("preserveAspectRatio", "xMinYMin");
        svg.call(zoom);
        var mainContainer = svg.append("g").attr("width", width).attr("height", height);

        var color = d3.scaleOrdinal(d3.schemeCategory20);

        var simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(function (d) {
                return d.id;
            }))
            .force("charge", d3.forceManyBody())
            .force("center", d3.forceCenter(width / 2, height / 2));

        if (error) throw error;

        var link = mainContainer.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(graph.links)
            .enter().append("line")
            .attr("stroke-width", function (d) {
                return d.value*2;
            });

        var node = mainContainer.append("g").attr("class", "nodes")
            .selectAll("nodes").data(graph.nodes).enter().append("g").attr("class","node");

        var text = node.append("text")
            .attr("dx", 10)
            .attr("dy", ".35em")
            .text(function (d) {
                return d.id;
            })
            .style("stroke", "gray");

        var circle = node
            .append("circle")
            .attr("r", 5)
            .attr("fill", function (d) {
                return color(d.group);
            })
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // circle.append("title")
        //     .text(function (d) {
        //         return d.id;
        //     });

        simulation
            .nodes(graph.nodes)
            .on("tick", ticked);

        simulation.force("link")
            .links(graph.links);

        function ticked() {
            link
                .attr("x1", function (d) {
                    return d.source.x;
                })
                .attr("y1", function (d) {
                    return d.source.y;
                })
                .attr("x2", function (d) {
                    return d.target.x;
                })
                .attr("y2", function (d) {
                    return d.target.y;
                });

            d3.selectAll("text")
                .attr("x", function (d) {
                    return d.x;
                })
                .attr("y", function (d) {
                    return d.y;
                });

            d3.selectAll("circle")
                .attr("cx", function (d) {
                    return d.x;
                })
                .attr("cy", function (d) {
                    return d.y;
                });
        }

        // zoom.scaleTo(svg, 2);

        $('.node').each(function () {
            $(this).children('text').hide();
            $(this).on('mouseover', function () {
                console.log('mouseover!')
                $(this).children('text').show();
            });
            $(this).on('mouseout', function () {
                $(this).children('text').hide();
            });
        })

        function zoomed() {
            mainContainer.attr("transform", d3.event.transform);
        }

        function dragstarted(d) {
            if (!d3.event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(d) {
            d.fx = d3.event.x;
            d.fy = d3.event.y;
        }

        function dragended(d) {
            if (!d3.event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    });



    return true;
}


$('#inputs').submit(loadData);