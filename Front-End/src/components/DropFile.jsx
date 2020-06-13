import React, { Component } from "react";
import "antd/dist/antd.css";
import { Upload, message } from "antd";
import InboxOutLined from "@ant-design/icons";
import { Tag, Table } from "antd";
import httpCommon from "../DataService/common-http";
import Spinner from "./Spinner";



const { Dragger } = Upload;

const columns = [
  {
    title: 'Index',
    dataIndex: 'id',
    key: 'id',
    render: id => <a>{id}</a>
  },
  {
    title: 'Content',
    dataIndex: 'content',
    key: 'content',
    render: text => <a>{text}</a>,
  },
  {
    title: 'Result',
    dataIndex: 'resultInBoolean',
    key: 'resultInBoolean',
    render: result => <Tag color={String(result) === 'true' ? 'green': 'red'} key = {String(result)}>
      {/* {String(String(result).toUpperCase())} */
          String(result) === 'true' ? 'Tích cực' : 'Tiêu cực'
      }
    </Tag>
  },
  {
    title: 'Value',
    dataIndex: 'resultInNumber',
    key: 'resultInNumber',
    render: result => <Tag color='blue' key = {String(result)}>
    {
        String(result)
    }
    </Tag>
  }
];

class DropFile extends Component {
  constructor(props) {
    super(props);

    this.state = {
      post: [],
      datas: [],
      isLoading: false
    };

    this.onChange = this.onChange.bind(this);
    this.fetchData = this.fetchData.bind(this);
  }

  onChange(info) {
    if (info.file.status !== "uploading") {
      let reader = new FileReader();
      reader.onload = e => {
        // console.log(e.target.result);
        // console.log(reader.result)
        var data = e.target.result;
        var lines = data.split("\n");
        var result = [];
        for (var i = 0; i < lines.length; i++) {
          var obj = {};
          var currentLine = String();
          currentLine = lines[i].split("\r");
          currentLine = currentLine.slice(0, currentLine.length - 1);
          var part2 = `"${currentLine}"`;
          obj[i] = JSON.parse(
            `{"content": ${part2}, "resultInNumber": 0.1, "resultInBoolean": false}`
          );
          result.push(obj[i]);
        }
        // console.log(post);
        this.setState({ post: result });
        console.log(this.state.post);
      };
      reader.readAsText(info.file.originFileObj, "UTF-8");
    }
    if (info.file.status === "done") {
      message.success(`${info.file.name} file uploaded successfully`);
    } else if (info.file.status === "error") {
      message.error(`${info.file.name} file upload failed.`);
    }
  }
  
  async fetchData() {
    this.setState({isLoading: true})
    var postToSend = this.state.post;
    // console.log(postToSend)
    // axios.post("http://localhost:8000/api/posts", {
    //   postToSend,
    // }, {
    //   headers: headers
    // })

    let {data} = await httpCommon(postToSend);

    console.log(data);

    this.setState({
      datas: data,
      isLoading: false
    });


    // dataService.postData(postToSend)
    //   .then(response => JSON.stringify(response))
    //   .then(res => {
    //     this.setState({
    //       datas: res
    //     });
    //     console.log(this.state.datas);
    //   })
    //   .catch(error => {
    //     console.log(error, "some error has happened");
    //   });
  }

  render() {
  
    const props = {
      name: "file",
      // action: "//jsonplaceholder.typicode.com/posts/",
      action: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
      headers: {
        authorization: "authorization-text",
      },
      multiple: true,
      accepted: ".txt"
    };
    return (
      <React.Fragment>
        <Dragger {...props} onChange={this.onChange} className="dragger">
          <p className="ant-'upload'-drag-icon">
            <InboxOutLined />
          </p>
          <p className="ant-upload-text"> Click or Drag file</p>
          <p className="ant-upload-hint"> Put some hints in here</p>
        </Dragger>
        
        {
          this.state.isLoading === true ? <Spinner/> :
          <React.Fragment>
            <button onClick={this.fetchData} className="custom-button"></button>
            <Table columns={columns} dataSource={this.state.datas} className="data-table" />
          </React.Fragment>
        }
      </React.Fragment>
    );
  }
}

export default DropFile;
