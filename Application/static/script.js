/**
 * Created by Zoe on 13/08/16.
 */

$('.datepicker-here').datepicker({
	maxDate: new Date(),
    dateFormat: "yyyy-mm-dd"
})

var loading;

function loadData() {
    // get values
    var results = $('#results').val();
    var startDate = $('#startDate').val();
    var endDate = $('#endDate').val();
    var clusters = $('#clusters').val();

    $('form').attr("style","display:none");

    $('body').prepend("<svg></svg>");

    loading = true;

    $("img.loading").attr("style","display: block;");

    var msgs = $(".msg");
    var msgIndex = -1;

    function showNextMsg() {
        if(loading==true){
            ++msgIndex;
            msgs.eq(msgIndex % msgs.length)
                .fadeIn(2000)
                .delay(2000)
                .fadeOut(2000, showNextMsg);
            }
    }

    showNextMsg();

    visualise(results, startDate, endDate, clusters);

    return false;
};

function visualise(results, startDate, endDate, clusters) {
    var json_source = $SCRIPT_ROOT + '/cluster?results=' + results + '&start_date=' + startDate + '&end_date=' + endDate + '&clusters=' + clusters
    // var json_source = $SCRIPT_ROOT + '/static/miserables.json'
    // json_source = $SCRIPT_ROOT + '/static/output.json'

    d3.json(json_source, function (error, graph) {

        $('body').append('<div id="sidePanel"></div>');
        $('#sidePanel').append('<div id="title"></div>');
        $('#sidePanel').append('<div id="features"></div>');
        $('#sidePanel').append('<div id="content"></div>');
        $('#sidePanel').append('<div id="rating"></div>');

        // $('#sidePanel').append('<div id="rating"></div>');

        var width = +window.innerWidth;
        var height = +window.innerHeight;
        var zoom = d3.zoom().scaleExtent([.2, 10]).on("zoom", zoomed);

        var svg = d3.select("svg").attr("viewBox", "0 0 " + width + " " + height ).attr("preserveAspectRatio", "xMinYMin");
        svg.call(zoom);
        var mainContainer = svg.append("g").attr("width", width).attr("height", height);

        var color = d3.scaleOrdinal(d3.schemeCategory20);

        var simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(function (d) {
                return d.id;
            }).distance(function(d){
                return Math.pow(d.value*10,1.1);   // increasing the distance a bit for clearer visuals
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
                return 30/(d.value);    // the higher the distance, the thinner the line
            });

        var node = mainContainer.append("g").attr("class", "nodes")
            .selectAll("nodes").data(graph.nodes).enter().append("g").attr("class","node");

        var text = node.append("text")
            .attr("dx", 20)
            .attr("dy", "0.5em")
            .text(function (d) {
                return d.id;
            })
            .style("stroke", "gray");

        var circle = node
            .append("circle")
            .attr("r", 13)
            .attr("fill", function (d) {
                return color(d.group);
            })
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on('click', function(d) {
                $('#title').html("<h1>" + d.id + "</h1>");
                $('#features').html(d.features);
                $('#content').html(d.bodyhtml);
                $('#rating').html(d.silhouette);
                // $('#rating').html(d.silhouette);
                
            });

        $('.node').each(function () {
            $(this).children('text').hide();
            $(this).on('mouseover', function () {
                $(this).children('text').show();
            });
            $(this).on('mouseout', function () {
                $(this).children('text').hide();
            });
        })

        loading = false;
        $('.loading').attr("style","display: none;");

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
